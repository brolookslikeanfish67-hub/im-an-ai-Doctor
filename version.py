# pylint: disable=missing-docstring
import argparse
import logging
import os
import subprocess

# Fallback version used if --git-describe is not invoked or fails.
# Two tag formats are supported:
# - vMAJ.MIN.PATCH (e.g. v0.8.0)
# - vMAJ.MIN.devN (e.g. v0.8.dev0)
__version__ = "0.1.dev0"
PROJ_ROOT = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


def git_describe_version():
    """Get PEP-440 compatible public and local version using git describe.

    Returns
    -------
    pub_ver: str or None
        Public version.
    local_ver: str or None
        Local version (with additional label appended to pub_ver).
    """
    cmd = [
        "git",
        "describe",
        "--tags",
        "--match", "v[0-9]*.[0-9]*.[0-9]*",
        "--match", "v[0-9]*.[0-9]*.dev[0-9]*",
    ]
    
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True, 
        cwd=PROJ_ROOT
    )

    if result.returncode != 0:
        logging.warning("git describe failed: %s", result.stdout.strip())
        return None, None

    describe = result.stdout.strip()
    arr_info = describe.split("-")

    # Normalize 'v' prefix
    if arr_info[0].startswith("v"):
        arr_info[0] = arr_info[0][1:]

    # Hit the exact tag
    if len(arr_info) == 1:
        return arr_info[0], arr_info[0]

    if len(arr_info) != 3:
        logging.warning("Unexpected output format from git describe: %s", describe)
        return None, None

    # Handle dev or patch splits
    base_version = arr_info[0].split(".dev")[0]

    pub_ver = f"{base_version}.dev{arr_info[1]}"
    local_ver = f"{pub_ver}+{arr_info[2]}"
    
    return pub_ver, local_ver


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Detect and synchronize version.")
    parser.add_argument(
        "--print-version",
        action="store_true",
        help="Print version to the command line. No changes are applied to files.",
    )
    parser.add_argument(
        "--git-describe",
        action="store_true",
        help="Use git describe to generate development version.",
    )
    parser.add_argument("--dry-run", action="store_true")
    
    opt = parser.parse_args()
    
    pub_ver, local_ver = None, None
    if opt.git_describe:
        pub_ver, local_ver = git_describe_version()
        
    # Fallback to defaults if git execution wasn't requested or failed
    if pub_ver is None:
        pub_ver = __version__
    if local_ver is None:
        local_ver = __version__
        
    if opt.print_version:
        print(local_ver)


if __name__ == "__main__":
    main()
