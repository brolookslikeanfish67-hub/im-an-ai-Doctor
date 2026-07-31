# pylint: disable=invalid-name,missing-docstring,too-many-locals
import argparse
import json
import os
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import tvm
from transformers import AutoTokenizer, LlamaTokenizer  # type: ignore[import]
from tvm import relax
from tvm.relax.testing.lib_comparator import LibCompareVMInstrument
from tvm.runtime import ShapeTuple

from mlc_llm import utils


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLC-LLM Execution Profiler")
    parser.add_argument("--local-id", type=str, required=True)
    parser.add_argument("--device-name", type=str, default="auto")
    parser.add_argument("--debug-dump", action="store_true", default=False)
    parser.add_argument("--artifact-path", type=str, default="dist")
    parser.add_argument("--prompt", type=str, default="The capital of Canada is")
    parser.add_argument("--profile", action="store_true", default=False)
    
    parsed = parser.parse_args()
    parsed.model, parsed.quantization = parsed.local_id.rsplit("-", 1)
    utils.argparse_postproc_common(parsed)
    
    parsed.artifact_path = os.path.join(
        parsed.artifact_path, f"{parsed.model}-{parsed.quantization.name}"
    )
    return parsed


class LibCompare(LibCompareVMInstrument):
    def __init__(self, mod: tvm.runtime.Module, device: tvm.runtime.Device):
        super().__init__(mod, device, verbose=False)
        self.time_eval_results: Dict[str, Tuple[float, int, List[Tuple[int, ...]], int]] = {}

    def compare(
        self,
        name: str,
        ref_args: List[tvm.nd.NDArray],
        new_args: List[tvm.nd.NDArray],
        ret_indices: List[int],
    ) -> None:
        if name.startswith("shape_func"):
            return
            
        if name not in self.time_eval_results:
            super().compare(name, ref_args, new_args, ret_indices)
            
            # Execute profile timing loop on target device framework
            mean_time = self.mod.time_evaluator(
                name,
                dev=self.device,
                number=100,
                repeat=3,
            )(*new_args).mean
            
            shapes = [tuple(arg.shape) for arg in new_args]
            
            # Zero-copy memory footprint extraction using native backing data descriptors
            total_bytes = sum(
                arg.data.size * np.dtype(arg.dtype).itemsize for arg in new_args
            )
            
            self.time_eval_results[name] = (mean_time, 1, shapes, total_bytes)
        else:
            record = self.time_eval_results[name]
            self.time_eval_results[name] = (
                record[0],
                record[1] + 1,
                record[2],
                record[3],
            )


def print_as_table(sorted_list: List[Tuple[str, Tuple[float, int, List[Tuple[int, ...]], int]]]) -> None:
    if not sorted_list:
        print("No profiling results recorded.\n")
        return

    # Header with modern field alignments
    print(
        f"{'Name':<50}{'Time (ms)':<12}{'Count':<8}{'Total (ms)':<18}"
        f"{'Pct (%)':<10}{'Memory (MB)':<16}{'BW (GB/s)':<18}{'Shape'}"
    )
    
    total_time = sum(record[1][0] * record[1][1] for record in sorted_list) * 1000
    if total_time == 0:
        return

    for record in sorted_list:
        name = record[0]
        mean_sec, count, shapes, total_bytes = record[1]
        
        time_ms = mean_sec * 1000
        weighted_time = time_ms * count
        percentage = (weighted_time / total_time) * 100
        
        memory_mb = total_bytes / (1024 * 1024)
        bandwidth = (total_bytes / mean_sec / (1024**3)) if mean_sec > 0 else 0.0
        shape_str = ", ".join(str(s) for s in shapes)

        print(
            f"{name:<50}"
            f"{time_ms:<12.4f}"
            f"{str(count):<8}"
            f"{weighted_time:<18.4f}"
            f"{percentage:<10.2f}"
            f"{memory_mb:<16.2f}"
            f"{bandwidth:<18.4f}"
            f"{shape_str}"
        )
    print(f"Total time: {total_time:.4f} ms\n")


def deploy_to_pipeline(args: argparse.Namespace) -> None:
    device = tvm.device(args.device_name)
    const_params = utils.load_params(args.artifact_path, device)
    
    model_lib_path = os.path.join(
        args.artifact_path,
        f"{args.model}-{args.quantization.name}-{args.device_name}.so",
    )
    ex = tvm.runtime.load_module(model_lib_path)
    vm = relax.VirtualMachine(ex, device)

    config_path = os.path.join(args.artifact_path, "params", "mlc-chat-config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    tokenizer_path = os.path.join(args.artifact_path, "params")
    if config.get("model_category") == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    print("Tokenizing...")
    inputs = tvm.nd.array(
        tokenizer(args.prompt, return_tensors="pt").input_ids.to(torch.int32).numpy(),
        device,
    )
    first_sampled_token = tvm.nd.array(np.array([[6234]], dtype="int32"), device)
    seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1]])
    second_seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1] + 1])
    
    # Warmup Run to clear dynamic allocation noise
    kv_caches = vm["create_kv_cache"]()
    logits, kv_caches = vm["prefill"](inputs, seq_len_shape, kv_caches, const_params)
    logits, kv_caches = vm["decode"](
        first_sampled_token, second_seq_len_shape, kv_caches, const_params
    )
    device.sync()

    # Clear state caches and run real benchmark loop
    kv_caches = vm["create_kv_cache"]()
    print("Running inference...")
    
    start = time.time()
    logits, kv_caches = vm["prefill"](inputs, seq_len_shape, kv_caches, const_params)
    device.sync()
    encoding_end = time.time()
    
    logits, kv_caches = vm["decode"](
        first_sampled_token, second_seq_len_shape, kv_caches, const_params
    )
    device.sync()
    end = time.time()
    
    if args.debug_dump:
        fcache_view = tvm.get_global_func("vm.builtin.attention_kv_cache_view")
        first_k_cache = fcache_view(kv_caches[0], ShapeTuple([7, 32, 128]))
        print(f"output kv_cache[0]:\n{first_k_cache.numpy().transpose(1, 0, 2)}")
        print(f"output logits:\n{logits.numpy()}")
        
    print(
        f"Time elapsed: encoding {(encoding_end - start):.4f} seconds, "
        f"decoding {(end - encoding_end):.4f} secs"
    )

    if args.profile:
        cmp_instrument = LibCompare(ex, device)
        vm.set_instrument(cmp_instrument)

        print("Profiling Pipeline...")
        kv_caches = vm["create_kv_cache"]()

        # Profile Phase 1: Prefill Stage
        logits, kv_caches = vm["prefill"](inputs, seq_len_shape, kv_caches, const_params)
        print("======================= Encoding Profiling =======================")
        print_as_table(sorted(cmp_instrument.time_eval_results.items(), key=lambda x: -(x[1][0] * x[1][1])))
        cmp_instrument.time_eval_results.clear()

        # Profile Phase 2: Decode Stage
        logits, kv_caches = vm["decode"](
            first_sampled_token, second_seq_len_shape, kv_caches, const_params
        )
        print("======================= Decoding Profiling =======================")
        print_as_table(sorted(cmp_instrument.time_eval_results.items(), key=lambda x: -(x[1][0] * x[1][1])))


if __name__ == "__main__":
    ARGS = _parse_args()
    deploy_to_pipeline(ARGS)
