# pylint: disable=invalid-name,missing-docstring
import argparse
import json
import os
import unittest
from unittest.mock import MagicMock, mock_open, patch
from mlc_llm import utils
from mlc_llm.core import build_model_from_args

# --- MOCK DESIGN PATTERNS ---

class HighAvailabilityMockArgs(argparse.Namespace):
    """Factory providing baseline default configurations for build pipeline testing."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quantization = utils.quantization_schemes.get("q8f16_1")
        self.debug_dump = False
        self.use_cache = False
        self.sep_embed = False
        self.build_model_only = True
        self.use_safetensors = False
        self.convert_weight_only = False
        self.no_cutlass_attn = True
        self.no_cutlass_norm = True
        self.reuse_lib = True
        self.artifact_path = "/tmp/mlc_build/artifacts"
        self.model_path = "/tmp/mlc_build/model"
        self.model = "/tmp/mlc_build/base"
        self.target_kind = "cuda"
        self.max_seq_len = 2048
        self.model_category = "base"


# --- CONFIGURATION FIXTURES MATRIX ---

MODEL_TEST_MATRIX = {
    "llama": {
        "model_name": "/tmp/",
        "config_payload": {}
    },
    "gpt_neox": {
        "model_name": "dolly-test",
        "config_payload": {
            "use_parallel_residual": False,
            "hidden_size": 32,
            "intermediate_size": 32,
            "num_attention_heads": 32,
            "num_hidden_layers": 28,
            "vocab_size": 1024,
            "rotary_pct": 1,
            "rotary_emb_base": 1,
            "layer_norm_eps": 1,
        }
    },
    "gpt_bigcode": {
        "model_name": "gpt_bigcode",
        "config_payload": {}
    },
    "minigpt": {
        "model_name": "minigpt4-7b",
        "config_payload": {}
    },
    "gptj": {
        "model_name": "gpt-j-",
        "config_payload": {
            "vocab_size": 1024,
            "n_embd": 32,
            "n_inner": 32,
            "n_head": 32,
            "n_layer": 28,
            "bos_token_id": 28,
            "eos_token_id": 1,
            "rotary_dim": 1,
            "tie_word_embeddings": 1,
        }
    },
    "rwkv": {
        "model_name": "rwkv-",
        "config_payload": {
            "num_hidden_layers": 16,
            "vocab_size": 1024,
            "hidden_size": 16,
            "intermediate_size": 32,
        }
    },
    "chatglm": {
        "model_name": "chatglm2",
        "config_payload": {}
    }
}

# --- TEST SUITE CORNERSTONE ---

class CoreBuildModelPipelineTest(unittest.TestCase):

    def setUp(self):
        # Graceful sandbox patching instead of raw assignment overwrites
        self.mkdir_patcher = patch.object(os, "mkdir", MagicMock())
        self.mock_mkdir = self.mkdir_patcher.start()
        self.mock_args = HighAvailabilityMockArgs()

    def tearDown(self):
        self.mkdir_patcher.stop()

    def _execute_pipeline_verification(self, category: str, matrix_data: dict):
        """Internal runner to execute and check compilation lifecycle states."""
        self.mock_args.model_category = category
        self.mock_args.model = matrix_data["model_name"]

        with patch("builtins.open", mock_open(read_data="data")):
            with patch("json.load", MagicMock(return_value=matrix_data["config_payload"])):
                try:
                    build_model_from_args(self.mock_args)
                except Exception as error:
                    self.fail(f"Pipeline execution failed for target category '{category}': {error}")

    # --- ARCHITECTURE AGNOSTIC WRAPPERS ---

    def test_llama_model_compilation(self):
        self._execute_pipeline_verification("llama", MODEL_TEST_MATRIX["llama"])

    def test_gpt_neox_model_compilation(self):
        self._execute_pipeline_verification("gpt_neox", MODEL_TEST_MATRIX["gpt_neox"])

    def test_gpt_bigcode_model_compilation(self):
        self._execute_pipeline_verification("gpt_bigcode", MODEL_TEST_MATRIX["gpt_bigcode"])

    def test_minigpt_model_compilation(self):
        self._execute_pipeline_verification("minigpt", MODEL_TEST_MATRIX["minigpt"])

    def test_gptj_model_compilation(self):
        self._execute_pipeline_verification("gptj", MODEL_TEST_MATRIX["gptj"])

    def test_rwkv_model_compilation(self):
        self._execute_pipeline_verification("rwkv", MODEL_TEST_MATRIX["rwkv"])

    def test_chatglm_model_compilation(self):
        self._execute_pipeline_verification("chatglm", MODEL_TEST_MATRIX["chatglm"])


if __name__ == "__main__":
    unittest.main()
