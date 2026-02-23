"""
Self-Evolving VLM: From Self-Rewarding to Self-Evolving
A Unified Framework for Adaptive Multimodal Reasoning

Setup script for installation.
"""

from setuptools import setup, find_packages

setup(
    name="self-evolving-vlm",
    version="0.1.0",
    description="A Unified Framework for Adaptive Multimodal Reasoning (HSR-EE + IPR + SR-DPO)",
    author="SR2SE Team",
    python_requires=">=3.9",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.37.0",
        "accelerate>=0.25.0",
        "datasets>=2.16.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "jsonlines>=4.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "train": [
            "trl>=0.7.0",
            "peft>=0.7.0",
            "bitsandbytes>=0.41.0",
            "deepspeed>=0.12.0",
            "wandb>=0.16.0",
            "tensorboard>=2.15.0",
        ],
        "eval": [
            "evalscope>=0.4.0",
        ],
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "isort>=5.12",
        ],
    },
    entry_points={
        "console_scripts": [
            "sr2se-pipeline=scripts.run_pipeline:main",
            "sr2se-infer=scripts.inference:main",
        ],
    },
)
