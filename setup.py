import os
from setuptools import setup, find_packages
from setuptools.dist import Distribution

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def git_describe_version(original_version):
    """Get git describe version cleanly from version.py."""
    ver_py = os.path.join(CURRENT_DIR, "mlc_llm", "version.py")
    
    if not os.path.exists(ver_py):
        # Fallback if version.py isn't in the subfolder
        ver_py = os.path.join(CURRENT_DIR, "version.py")
        
    libver = {}
    try:
        with open(ver_py, "rb") as f:
            exec(compile(f.read(), ver_py, "exec"), libver)
        _, gd_version = libver["git_describe_version"]()
        
        if gd_version and gd_version != original_version:
            print(f"Use git describe based version: {gd_version}")
        return gd_version
    except (FileNotFoundError, KeyError):
        return original_version or "0.0.0"

__version__ = git_describe_version(None)

setup(
    name="mlc_llm",
    version=__version__,
    description="MLC LLM: Universal Compilation of Large Language Models",
    url="https://mlc.ai/mlc-llm/",
    author="MLC LLM Contributors",
    license="Apache 2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
    keywords="machine learning llm compilation",
    zip_safe=False,
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    install_requires=[
        "numpy",
        "torch",
        "transformers",
        "scipy",
        "timm",
    ],
    entry_points={
        "console_scripts": [
            "mlc_llm_build = mlc_llm.build:main",
        ],
    },
    distclass=Distribution,
)
