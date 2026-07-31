import argparse
import json
import os
from typing import List

import numpy as np
import torch
import tvm
from mlc_llm import utils
from transformers import AutoTokenizer, LlamaTokenizer
from tvm import relax, rpc
from tvm.relax.testing.lib_comparator import LibCompareVMInstrument


class LibCompare(LibCompareVMInstrument):
    def __init__(self, mod, device, time_eval, skip_rounds=0):
        super().__init__(mod, device, True)
        self.time_eval = time_eval
        self.time_eval_results = {}
        self.visited = set()
        self.skip_rounds = skip_rounds
        self.atol = 1e-2
        self.rtol = 1e-3

    def skip_instrument(self, func, name, before_run, ret_val, *args):
        print(f"run {name}")
        if name.startswith("shape_func"):
            return True
        if self.counter < self.skip_rounds:
            self.counter += 1
            print(f"[{self.counter}] Skip validating {name}..")
            return True
        if name in self.visited:
            if self.time_eval and name in self.time_eval_results:
                record = self.time_eval_results[name]
                self.time_eval_results[name] = (record[0], record[1] + 1)
            return True
        self.visited.add(name)
        return False

    def compare(
        self,
        name: str,
        ref_args: List[tvm.nd.NDArray],
        new_args: List[tvm.nd.NDArray],
        ret_indices: List[int],
    ):
        super().compare(name, ref_args, new_args, ret_indices)

        if self.time_eval and name not in self.time_eval_results:
            res = self.mod.time_evaluator(
                name, self.device, number=20, repeat=3
            )(*new_args)
            self.time_eval_results[name] = (res.mean, 1)
            print(f"Time-eval result {name} on {self.device}: {res}")


def print_as_table(sorted_list):
    if not sorted_list:
        print("No profiling results to display.\n")
        return

    # Header using modern alignment
    print(
        f"{'Name':<50}{'Time (ms)':<12}{'Count':<8}{'Total time (ms)':<18}{'Percentage (%)'}"
    )
    
    total_time = sum(record[1][0] * record[1][1] for record in sorted_list) * 1000
    if total_time == 0:
        return

    for record in sorted_list:
        time_ms = record[1][0] * 1000
        weighted_time = time_ms * record[1][1]
        percentage = (weighted_time / total_time) * 100
        
        print(
            f"{record[0]:<50}"
            f"{time_ms:<12.4f}"
            f"{str(record[1][1]):<8}"
            f"{weighted_time:<18.4f}"
            f"{percentage:.2f}"
        )
    print(f"Total time: {total_time:.4f} ms\n")


class TestState:
    def __init__(self, args):
        self.primary_device = tvm.device(args.primary_device)
        ex = tvm.runtime.load_module(
            os.path.join(
                args.artifact_path,
                f"{args.model}-{args.quantization.name}-{args.primary_device}.so",
            )
        )
        self.vm = relax.VirtualMachine(ex, self.primary_device)
        
        # Cross-compilation target setup
        if args.cmp_device == "iphone":
            lib_name = f"{args.model}-{args.quantization.name}-{args.cmp_device}.dylib"
            local_lib_path = os.path.join(args.artifact_path, lib_name)
            proxy_host = os.environ.get("TVM_RPC_PROXY_HOST", "127.0.0.1")
            proxy_port = int(os.environ.get("TVM_RPC_PROXY_PORT", "9090"))
            self.sess = rpc.connect(proxy_host, proxy_port, "iphone")
            self.sess.upload(local_lib_path)
            self.lib = self.sess.load_module(lib_name)
            self.cmp_device = self.sess.metal()
        elif args.cmp_device == "android":
            lib_name = f"{args.model}-{args.quantization.name}-{args.cmp_device}.so"
            local_lib_path = os.path.join(args.artifact_path, lib_name)
            tracker_host = os.environ.get("TVM_TRACKER_HOST", "0.0.0.0")
            tracker_port = int(os.environ.get("TVM_TRACKER_PORT", "9190"))
            tracker = rpc.connect_tracker(tracker_host, tracker_port)
            self.sess = tracker.request("android")
            self.sess.upload(local_lib_path)
            self.lib = self.sess.load_module(lib_name)
            self.cmp_device = self.sess.cl(0)
        else:
            self.sess = None
            self.lib = tvm.runtime.load_module(
                os.path.join(
                    args.artifact_path,
                    f"{args.model}-{args.quantization.name}-{args.cmp_device}.so",
                )
            )
            self.cmp_device = tvm.device(args.cmp_device)

        self.cmp_instrument = LibCompare(
            self.lib,
            self.cmp_device,
            time_eval=args.time_eval,
            skip_rounds=args.skip_rounds,
        )
        self.vm.set_instrument(self.cmp_instrument)


def deploy_to_pipeline(args) -> None:
    config_path = os.path.join(args.artifact_path, "params", "mlc-chat-config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    primary_device = tvm.device(args.primary_device)
    const_params = utils.load_params(args.artifact_path, primary_device)
    state = TestState(args)

    tokenizer_path = os.path.join(args.artifact_path, "params")
    if config.get("model_category") == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    print("Tokenizing...")
    inputs = tvm.nd.array(
        tokenizer(args.prompt, return_tensors="pt").input_ids.to(torch.int32).numpy(),
        primary_device,
    )
    first_sampled_token = tvm.nd.array(
        np.array([[6234]], dtype="int32"), primary_device
    )
    seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1]])
    second_seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1] + 1])
    kv_caches = state.vm["create_kv_cache"]()

    print("Running inference...")
    print("======================= Starts Encoding =======================")
    logits, kv_caches = state.vm["prefill"](
        inputs, seq_len_shape, kv_caches, const_params
    )
    print_as_table(
        sorted(
            state.cmp_instrument.time_eval_results.items(),
            key=lambda x: -(x[1][0] * x[1][1]),
        )
    )
    state.cmp_instrument.time_eval_results.clear()
    state.cmp_instrument.visited.clear()
    
    print("======================= Starts Decoding =======================")
    logits, kv_caches = state.vm["decode"](
        first_sampled_token, second_seq_len_shape, kv_caches, const_params
    )
    print_as_table(
        sorted(
            state.cmp_instrument.time_eval_results.items(),
            key=lambda x: -(x[1][0] * x[1][1]),
        )
    )
    state.cmp_instrument.time_eval_results.clear()


def _parse_args():
    parser = argparse.ArgumentParser(description="Cross-Device Layer-by-Layer Verification")
    parser.add_argument("--local-id", type=str, required=True)
    parser.add_argument("--artifact-path", type=str, default="dist")
    parser.add_argument("--primary-device", type=str, default="auto")
    parser.add_argument("--cmp-device", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The capital of Canada is")
    parser.add_argument("--time-eval", default=False, action="store_true")
    parser.add_argument("--skip-rounds", type=int, default=0)
    
    parsed = parser.parse_args()
    parsed.model, parsed.quantization = parsed.local_id.rsplit("-", 1)
    utils.argparse_postproc_common(parsed)

    parsed.artifact_path = os.path.join(
        parsed.artifact_path, f"{parsed.model}-{parsed.quantization.name}"
    )

    if parsed.primary_device == "auto":
        if tvm.cuda().exist:
            parsed.primary_device = "cuda"
        elif tvm.metal().exist:
            parsed.primary_device = "metal"
        elif tvm.rocm().exist:
            parsed.primary_device = "rocm"
        else:
            raise ValueError("Cannot auto-deduce baseline device. Please set --primary-device explicitly.")
    return parsed


if __name__ == "__main__":
    args = _parse_args()
    deploy_to_pipeline(args)
