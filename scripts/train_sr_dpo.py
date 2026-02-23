"""
Script: Train SR-DPO on IPR-generated preference data.

Usage:
    python scripts/train_sr_dpo.py \
        --model_path ./output/sft_ckpt \
        --preference_data ./output/ipr_data/preference_pairs.jsonl \
        --output_dir ./output/sr_dpo_ckpt \
        --beta 0.1 \
        --learning_rate 5e-7 \
        --batch_size 64 \
        --num_epochs 3
"""

import argparse
import logging
import sys
import os
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sr_dpo.trainer import SRDPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train SR-DPO")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the SFT checkpoint (policy model)")
    parser.add_argument("--preference_data", type=str, required=True,
                        help="Path to IPR preference data (JSONL)")
    parser.add_argument("--output_dir", type=str, default="./output/sr_dpo_ckpt",
                        help="Output directory for DPO model")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO regularization parameter beta")
    parser.add_argument("--learning_rate", type=float, default=5e-7,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    args = parser.parse_args()
    
    # Load policy model
    logger.info(f"Loading policy model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Create reference model (frozen copy)
    logger.info("Creating reference model (frozen copy of SFT checkpoint)...")
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Initialize SR-DPO trainer
    trainer = SRDPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        beta=args.beta,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )
    
    # Run training
    trainer.train(args.preference_data)
    
    logger.info(f"SR-DPO training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
