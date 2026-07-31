"""For testing the functionality of `BuildArgs` and `convert_build_args_to_argparser`."""

import argparse
import dataclasses
import unittest
from mlc_llm import BuildArgs, core, utils


def old_make_args() -> argparse.ArgumentParser:
    """The exact legacy method for generating the baseline ArgumentParser structure."""
    args = argparse.ArgumentParser()
    
    # 1. Structural String Configurations
    string_configs = {
        "--model": ("auto", 'The name of the model to build. If it is "auto", we will automatically set the model name according to "--model-path", "hf-path" or the model folders under "--artifact-path/models"'),
        "--hf-path": (None, "Hugging Face path from which to download params, tokenizer, and config"),
        "--reuse-lib": (None, "Whether to reuse a previously generated lib."),
        "--artifact-path": ("dist", "Where to store the output."),
        "--llvm-mingw": ("", "/path/to/llvm-mingw-root, use llvm-mingw to cross compile to windows."),
        "--target": ("auto", "The target platform to compile the model for."),
    }
    for flag, (default, help_text) in string_configs.items():
        args.add_argument(flag, type=str, default=default, help=help_text)

    # 2. Choice-restricted Configuration
    args.add_argument(
        "--quantization",
        type=str,
        choices=list(utils.quantization_schemes),
        default=list(utils.quantization_schemes)[0],
        help="The quantization mode we use to compile."
    )

    # 3. Numeric Configurations
    args.add_argument("--max-seq-len", type=int, default=-1, help="The maximum allowed sequence length for the model.")
    args.add_argument("--use-cache", type=int, default=1, help="Whether to use previously pickled IRModule and skip trace.")

    # 4. Flags / Boolean Actions
    boolean_flags = {
        "--debug-dump": "Whether to dump debugging files during compilation.",
        "--debug-load-script": "Whether to load the script for debugging.",
        "--system-lib": "A parameter to `relax.build`.",
        "--sep-embed": (
            "Build with separated embedding layer, only applicable to LlaMa. "
            "This feature is in testing stage, and will be formally replaced after "
            "massive overhaul of embedding feature for all models and use cases"
        ),
    }
    for flag, help_text in boolean_flags.items():
        args.add_argument(flag, action="store_true", default=False, help=help_text)

    return args


class BuildArgsTest(unittest.TestCase):
    """Tests whether BuildArgs reaches functional parity with regular ArgumentParser."""

    def argparsers_equal(self, parse_a: argparse.ArgumentParser, parse_b: argparse.ArgumentParser):
        """Helper matrix ensuring action schemas mirror each other exactly."""
        # pylint: disable=protected-access
        self.assertEqual(len(parse_a._actions), len(parse_b._actions))
        
        for x, y in zip(parse_a._actions, parse_b._actions):
            xx = {k: v for k, v in vars(x).items() if k != "container"}
            yy = {k: v for k, v in vars(y).items() if k != "container"}
            
            # Type execution comparison for dynamic types/choices
            if xx.get("choices") and yy.get("choices"):
                for expected_choice in yy["choices"] + xx["choices"]:
                    type_a = xx.get("type")
                    type_b = yy.get("type")
                    if callable(type_a) and callable(type_b):
                        self.assertEqual(type_a(expected_choice), type_b(expected_choice))
                xx.pop("type", None)
                yy.pop("type", None)

            self.assertEqual(xx, yy)
        # pylint: enable=protected-access

    def test_new_and_old_arg_parse_are_equivalent(self):
        """Validates that dynamically building the layout mirrors the explicit template."""
        self.argparsers_equal(core.convert_build_args_to_argparser(), old_make_args())

    def test_namespaces_are_equivalent_str(self):
        """Validates object generation vs traditional string-fed entry strings."""
        build_args = BuildArgs(model="RedPJ", target="cuda")
        build_args_namespace = argparse.Namespace(**dataclasses.asdict(build_args))

        empty_args = core.convert_build_args_to_argparser()
        parsed_args = empty_args.parse_args(["--model", "RedPJ", "--target", "cuda"])

        self.assertEqual(build_args_namespace, parsed_args)

        # Confirm assertion validation detects negative structural differences
        bad_build_args = BuildArgs(model="RedPJ", target="vulkan")
        bad_build_args_namespace = argparse.Namespace(**dataclasses.asdict(bad_build_args))
        self.assertNotEqual(bad_build_args_namespace, parsed_args)

    def test_namespaces_are_equivalent_str_boolean_int(self):
        """Ensures integrity over combined datatypes (integers, flags, strings)."""
        build_args = BuildArgs(model="RedPJ", max_seq_len=20, debug_dump=True)
        build_args_namespace = argparse.Namespace(**dataclasses.asdict(build_args))

        empty_args = core.convert_build_args_to_argparser()
        parsed_args = empty_args.parse_args(
            ["--model", "RedPJ", "--max-seq-len", "20", "--debug-dump"]
        )
        self.assertEqual(build_args_namespace, parsed_args)

        # Mismatched Boolean State Validation
        missing_bool_args = BuildArgs(model="RedPJ", max_seq_len=20)
        missing_bool_namespace = argparse.Namespace(**dataclasses.asdict(missing_bool_args))
        self.assertNotEqual(missing_bool_namespace, parsed_args)

        # Mismatched Integer Value Validation
        diff_int_args = BuildArgs(model="RedPJ", max_seq_len=18, debug_dump=True)
        diff_int_namespace = argparse.Namespace(**dataclasses.asdict(diff_int_args))
        self.assertNotEqual(diff_int_namespace, parsed_args)


if __name__ == "__main__":
    unittest.main()
