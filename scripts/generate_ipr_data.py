"""
Script: Generate IPR preference data.

Usage:
    python scripts/generate_ipr_data.py \
        --model_path ./output/sft_ckpt \
        --dataset_path path/to/dataset.jsonl \
        --output_dir ./output/ipr_data \
        --max_refinement_iterations 2 \
        --num_samples 50000
"""

import argparse
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import FrameworkConfig, IPRConfig
from src.models.vlm_wrapper import VLMWrapper
from src.ipr.data_engine import IPRDataEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(path: str):
    """Load dataset from JSONL or JSON file."""
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
    else:
        raise ValueError(f"Unsupported file format: {path}")
    
    logger.info(f"Loaded {len(data)} samples from {path}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Generate IPR preference data")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the SFT checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the training dataset (JSONL/JSON)")
    parser.add_argument("--output_dir", type=str, default="./output/ipr_data",
                        help="Output directory for preference data")
    parser.add_argument("--max_refinement_iterations", type=int, default=2,
                        help="Maximum refinement iterations (default: 2)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to process (default: all)")
    parser.add_argument("--device_map", type=str, default="auto",
                        help="Device map for model loading")
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    
    # Initialize VLM
    vlm = VLMWrapper(
        model_name_or_path=args.model_path,
        device_map=args.device_map,
    )
    
    # Configure IPR
    ipr_config = IPRConfig(
        max_refinement_iterations=args.max_refinement_iterations,
        num_samples=args.num_samples or len(dataset),
    )
    
    # Initialize IPR engine
    ipr_engine = IPRDataEngine(vlm=vlm, config=ipr_config)
    
    # Generate preference data
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "preference_pairs.jsonl")
    
    pairs, stats = ipr_engine.generate_preference_dataset(
        dataset=dataset,
        output_path=output_path,
        max_samples=args.num_samples,
    )
    
    # Save statistics
    stats_path = os.path.join(args.output_dir, "ipr_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "total_samples": stats.total_samples,
            "initially_correct": stats.initially_correct,
            "refined_success": stats.refined_success,
            "refined_failed": stats.refined_failed,
            "pairs_generated": stats.pairs_generated,
            "success_rate": stats.success_rate,
            "pair_yield": stats.pair_yield,
        }, f, indent=2)
    
    print(f"\n{stats}")
    print(f"Preference data saved to: {output_path}")
    print(f"Statistics saved to: {stats_path}")


if __name__ == "__main__":
    main()
