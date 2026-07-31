import argparse
import os
import numpy as np
import torch
import tvm
from tvm import relax
from transformers import AutoTokenizer
from mlc_llm import utils


class DumpInstrument:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.counter = 0

    def __call__(self, func, name, before_run, ret_val, *args):
        if before_run or name.startswith("vm.builtin."):
            return
        if any(not isinstance(x, tvm.nd.NDArray) for x in args):
            return

        if self.verbose:
            print(f"[{self.counter}][{name}]")
            print(args[-1])
        self.counter += 1


def print_as_table(sorted_list):
    # Header
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
        model_path = os.path.join(
            args.artifact_path,
            f"{args.model}-{args.quantization.name}-{args.primary_device}.so",
        )
        
        ex = tvm.runtime.load_module(model_path)
        self.vm = relax.VirtualMachine(ex, self.primary_device)
        self.instrument = DumpInstrument(verbose=True)
        self.vm.set_instrument(self.instrument)


def deploy_to_pipeline(args) -> None:
    primary_device = tvm.device(args.primary_device)
    const_params = utils.load_params(args.artifact_path, primary_device)
    state = TestState(args)
    
    tokenizer_path = os.path.join(args.artifact_path, "params")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    print("Tokenizing...")
    inputs = tokenizer(args.prompt, return_tensors="pt").input_ids.to(torch.int32).numpy()
    
    first_sampled_token = tvm.nd.array(np.array([[6234]], dtype="int32"), primary_device)
    
    kv_caches = state.vm["create_kv_cache"]()

    print("Running inference...")
    print("======================= Starts Encoding =======================")

    # Clean check if prefill exists in VM targets
    has_prefill = "prefill" in [f.name_hint for f in state.vm.module.functions]

    if inputs.shape[1] > 1 and has_prefill:
        inputs_nd = tvm.nd.array(inputs, device=primary_device)
        seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1]])
        logits, kv_caches = state.vm["prefill"](inputs_nd, seq_len_shape, kv_caches, const_params)
    else:
        # Fallback block processing sequence token by token
        for i in range(inputs.shape[1]):
            input_slice = tvm.nd.array(inputs[:, i : i + 1], device=primary_device)
            slice_shape = tvm.runtime.ShapeTuple([i + 1])
            logits, kv_caches = state.vm["decode"](
                input_slice, slice_shape, kv_caches, const_params
            )

    print("======================= Starts Decoding =======================")
    second_seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1] + 1])
    logits, kv_caches = state.vm["decode"](
        first_sampled_token, second_seq_len_shape, kv_caches, const_params
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="MLC-LLM Pipeline Runner")
    parser.add_argument("--local-id", type=str, required=True)
    parser.add_argument("--artifact-path", type=str, default="dist")
    parser.add_argument("--primary-device", type=str, default="auto")
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
        else:
            raise ValueError("Cannot auto-deduce target hardware device context. Please set --primary-device manually.")
            
    return parsed


if __name__ == "__main__":
    args = _parse_args()
    deploy_to_pipeline(args)
