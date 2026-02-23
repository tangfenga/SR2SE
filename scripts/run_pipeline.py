"""
Script: Run the complete Self-Evolving VLM pipeline (Algorithm 2).

Usage:
    python scripts/run_pipeline.py \
        --model_path Qwen/Qwen-VL-Chat \
        --dataset_path path/to/dataset.jsonl \
        --output_dir ./output \
        --skip_sft \
        --sft_checkpoint_path ./output/sft_ckpt
"""

import argparse
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import FrameworkConfig, ModelConfig, HSREEConfig, IPRConfig, SRDPOConfig
from src.training.pipeline import SelfEvolvingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


def load_dataset(path: str):
    """Load dataset from JSONL or JSON."""
    data = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Self-Evolving VLM: Full Training Pipeline"
    )
    # Model
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen-VL-Chat")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    
    # Pipeline control
    parser.add_argument("--skip_sft", action="store_true",
                        help="Skip SFT stage (use existing checkpoint)")
    parser.add_argument("--sft_checkpoint_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to JSON config file")
    
    # HSR-EE
    parser.add_argument("--confidence_threshold", type=float, default=0.7)
    parser.add_argument("--max_reasoning_steps", type=int, default=5)
    parser.add_argument("--length_penalty_lambda", type=float, default=0.1)
    
    # IPR
    parser.add_argument("--max_refinement_iterations", type=int, default=2)
    parser.add_argument("--num_ipr_samples", type=int, default=50000)
    
    # SR-DPO
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=3)
    
    args = parser.parse_args()
    
    # Build or load config
    if args.config_path:
        config = FrameworkConfig.load(args.config_path)
    else:
        config = FrameworkConfig(
            model=ModelConfig(model_name_or_path=args.model_path),
            hsr_ee=HSREEConfig(
                confidence_threshold=args.confidence_threshold,
                max_reasoning_steps=args.max_reasoning_steps,
                length_penalty_lambda=args.length_penalty_lambda,
            ),
            ipr=IPRConfig(
                max_refinement_iterations=args.max_refinement_iterations,
                num_samples=args.num_ipr_samples,
            ),
            sr_dpo=SRDPOConfig(
                beta=args.beta,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
            ),
            output_dir=args.output_dir,
            sft_output_dir=os.path.join(args.output_dir, "sft_ckpt"),
            ipr_data_dir=os.path.join(args.output_dir, "ipr_data"),
            hsr_ee_output_dir=os.path.join(args.output_dir, "hsr_ee_ckpt"),
            sr_dpo_output_dir=os.path.join(args.output_dir, "sr_dpo_ckpt"),
        )
    
    # Save config
    config_save_path = os.path.join(args.output_dir, "config.json")
    os.makedirs(args.output_dir, exist_ok=True)
    config.save(config_save_path)
    logger.info(f"Config saved to {config_save_path}")
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    logger.info(f"Loaded {len(dataset)} training samples")
    
    # Initialize and run pipeline
    pipeline = SelfEvolvingPipeline(config)
    pipeline.run_full_pipeline(
        dataset=dataset,
        skip_sft=args.skip_sft,
        sft_checkpoint_path=args.sft_checkpoint_path,
    )


if __name__ == "__main__":
    main()
