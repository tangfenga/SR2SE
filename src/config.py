"""
Configuration module for the Self-Evolving VLM Framework.

Defines all hyperparameters and settings described in the paper:
- HSR-EE: confidence threshold tau, max reasoning steps K, length penalty lambda
- IPR: max refinement iterations N_max
- SR-DPO: regularization beta, learning rate, batch size
- Training pipeline: SFT, IPR data construction, SR-DPO optimization
"""

from dataclasses import dataclass, field
from typing import Optional, List
import json
import os


@dataclass
class HSREEConfig:
    """Configuration for Hierarchical Self-Rewarding with Early Exit."""
    # Confidence threshold tau (initialized at 0.7, optimized via REINFORCE)
    confidence_threshold: float = 0.7
    # Maximum reasoning depth K
    max_reasoning_steps: int = 5
    # Length penalty lambda for compute budget
    length_penalty_lambda: float = 0.1
    # Hidden dimension for the confidence head MLP g_phi
    confidence_head_hidden_dim: int = 256
    # Whether to use adaptive threshold optimization via REINFORCE
    adaptive_threshold: bool = True
    # REINFORCE learning rate for threshold optimization
    threshold_lr: float = 1e-3


@dataclass
class IPRConfig:
    """Configuration for Iterative Perception Refinement."""
    # Maximum refinement iterations N_max (paper default: 2)
    max_refinement_iterations: int = 2
    # Refinement prompt template
    refinement_prompt_template: str = (
        "The description '{perception}' was insufficient to answer the question "
        "correctly. Please re-examine the image. Focus on spatial relationships "
        "and small details that were missed. Provide a refined description."
    )
    # Whether to discard samples that fail all refinement attempts
    discard_failed_samples: bool = True
    # Number of samples for IPR data generation
    num_samples: int = 50000


@dataclass
class SRDPOConfig:
    """Configuration for Self-Rewarding Direct Preference Optimization."""
    # DPO regularization parameter beta
    beta: float = 0.1
    # Learning rate for SR-DPO optimization
    learning_rate: float = 5e-7
    # Batch size
    batch_size: int = 64
    # Number of training epochs for SR-DPO
    num_epochs: int = 3
    # Gradient accumulation steps
    gradient_accumulation_steps: int = 4
    # Maximum sequence length
    max_seq_length: int = 2048
    # Warmup ratio
    warmup_ratio: float = 0.1
    # Weight decay
    weight_decay: float = 0.01
    # Max gradient norm for clipping
    max_grad_norm: float = 1.0


@dataclass
class SFTConfig:
    """Configuration for Supervised Fine-Tuning stage."""
    # Number of SFT epochs
    num_epochs: int = 1
    # Learning rate for SFT
    learning_rate: float = 5e-5
    # Batch size
    batch_size: int = 4
    # Gradient accumulation steps
    gradient_accumulation_steps: int = 4
    # SFT dataset name (LLaVA-Instruct-150k)
    dataset_name: str = "llava_instruct_150k"
    # Maximum sequence length
    max_seq_length: int = 2048


@dataclass
class ModelConfig:
    """Configuration for the base VLM."""
    # Model name or path (Qwen-VL-Chat 7B as per paper)
    model_name_or_path: str = "Qwen/Qwen-VL-Chat"
    # Whether to use flash attention
    use_flash_attention: bool = True
    # Whether to use bf16
    bf16: bool = True
    # Whether to use gradient checkpointing
    gradient_checkpointing: bool = True
    # LoRA rank (0 = full fine-tuning)
    lora_rank: int = 0
    # LoRA alpha
    lora_alpha: int = 16
    # Device map strategy
    device_map: str = "auto"


@dataclass
class FrameworkConfig:
    """Master configuration for the entire Self-Evolving Framework."""
    model: ModelConfig = field(default_factory=ModelConfig)
    hsr_ee: HSREEConfig = field(default_factory=HSREEConfig)
    ipr: IPRConfig = field(default_factory=IPRConfig)
    sr_dpo: SRDPOConfig = field(default_factory=SRDPOConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    
    # Output directories
    output_dir: str = "./output"
    sft_output_dir: str = "./output/sft_ckpt"
    ipr_data_dir: str = "./output/ipr_data"
    hsr_ee_output_dir: str = "./output/hsr_ee_ckpt"
    sr_dpo_output_dir: str = "./output/sr_dpo_ckpt"
    eval_output_dir: str = "./output/eval_results"
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    seed: int = 42
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=lambda o: o.__dict__)
    
    @classmethod
    def load(cls, path: str) -> "FrameworkConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        config = cls()
        config.model = ModelConfig(**data.get("model", {}))
        config.hsr_ee = HSREEConfig(**data.get("hsr_ee", {}))
        config.ipr = IPRConfig(**data.get("ipr", {}))
        config.sr_dpo = SRDPOConfig(**data.get("sr_dpo", {}))
        config.sft = SFTConfig(**data.get("sft", {}))
        for key in ["output_dir", "sft_output_dir", "ipr_data_dir",
                     "hsr_ee_output_dir", "sr_dpo_output_dir", "eval_output_dir",
                     "logging_steps", "save_steps", "eval_steps", "seed"]:
            if key in data:
                setattr(config, key, data[key])
        return config
