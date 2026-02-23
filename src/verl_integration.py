"""
veRL Integration for SR-DPO Training.

This module provides integration with the veRL (Volcano Engine RL) framework
for efficient distributed SR-DPO training, as described in the paper's
implementation details.

Usage with veRL:
    python -m verl.trainer.main_ray \
        --config configs/verl_sr_dpo.yaml
"""

import os
import sys
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sr_dpo_reward_function(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict] = None,
) -> float:
    """
    Reward function compatible with veRL's RLHF interface.
    
    Computes R_HSR = r_visual + r_ans - lambda * k
    
    This function is designed to be plugged into veRL's reward computation
    pipeline for the SR-DPO training stage.
    
    Args:
        data_source: Dataset identifier.
        solution_str: Model's generated response (perception + reasoning + answer).
        ground_truth: Expected answer.
        extra_info: Additional info including:
            - 'num_reasoning_steps': k
            - 'perception': visual perception text
            - 'r_visual': visual grounding score
    
    Returns:
        HSR reward scalar.
    """
    lambda_penalty = 0.1  # Paper default
    
    # Parse solution
    answer = solution_str.strip().split("Final Answer:")[-1].strip() \
        if "Final Answer:" in solution_str else solution_str.strip()
    
    # r_ans = I(a == a_gt)
    gt = ground_truth.strip().lower()
    pred = answer.strip().lower()
    r_ans = 1.0 if (gt == pred or gt in pred or (len(gt) == 1 and pred.startswith(gt))) else 0.0
    
    # r_visual (from extra_info or default)
    r_visual = 0.0
    k = 1
    if extra_info:
        r_visual = extra_info.get("r_visual", 0.0)
        k = extra_info.get("num_reasoning_steps", 1)
    
    # R_HSR = r_visual + r_ans - lambda * k
    reward = r_visual + r_ans - lambda_penalty * k
    return reward


def sr_dpo_preference_reward(
    chosen_response: str,
    rejected_response: str,
    question: str,
    ground_truth: str,
    beta: float = 0.1,
) -> Dict[str, float]:
    """
    Compute preference-based reward for veRL's DPO interface.
    
    Args:
        chosen_response: y_w (refined perception).
        rejected_response: y_l (original perception).
        question: Query.
        ground_truth: Expected answer.
        beta: DPO regularization parameter.
    
    Returns:
        Dictionary with chosen/rejected rewards and margin.
    """
    # Simple heuristic reward: chosen should be closer to ground truth
    chosen_score = _text_similarity(chosen_response, ground_truth)
    rejected_score = _text_similarity(rejected_response, ground_truth)
    
    return {
        "chosen_reward": chosen_score,
        "rejected_reward": rejected_score,
        "reward_margin": chosen_score - rejected_score,
    }


def _text_similarity(text1: str, text2: str) -> float:
    """Simple word overlap similarity."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    overlap = len(words1 & words2)
    return overlap / max(len(words1), len(words2))


# =====================================================
# veRL Config Generator
# =====================================================

def generate_verl_config(
    model_path: str,
    preference_data_path: str,
    output_dir: str,
    beta: float = 0.1,
    learning_rate: float = 5e-7,
    batch_size: int = 64,
    num_epochs: int = 3,
) -> Dict[str, Any]:
    """
    Generate a veRL-compatible configuration for SR-DPO training.
    
    Args:
        model_path: Path to the SFT checkpoint.
        preference_data_path: Path to IPR preference data.
        output_dir: Output directory.
        beta: DPO beta.
        learning_rate: Learning rate.
        batch_size: Batch size.
        num_epochs: Number of epochs.
    
    Returns:
        veRL configuration dictionary.
    """
    config = {
        "trainer": {
            "total_epochs": num_epochs,
            "project_name": "self-evolving-vlm",
            "experiment_name": "sr-dpo",
            "logger": ["console", "tensorboard"],
        },
        "data": {
            "train_files": preference_data_path,
            "max_prompt_length": 1024,
            "max_response_length": 1024,
        },
        "actor_rollout_ref": {
            "model": {
                "path": model_path,
            },
            "actor": {
                "strategy": "fsdp",
                "optim": {
                    "lr": learning_rate,
                    "weight_decay": 0.01,
                },
                "ppo_mini_batch_size": batch_size,
                "ppo_micro_batch_size_per_gpu": batch_size // 8,
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": batch_size // 8,
            },
        },
        "algorithm": {
            "kl_ctrl": {
                "kl_coef": beta,
            },
        },
        "reward_model": {
            "reward_fn": "src.verl_integration.sr_dpo_reward_function",
        },
    }
    
    return config
