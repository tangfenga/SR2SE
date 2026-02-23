"""
Script: Run inference with the trained Self-Evolving VLM.

Usage:
    python scripts/inference.py \
        --model_path ./output/sr_dpo_ckpt/final \
        --gate_path ./output/hsr_ee_ckpt/confidence_head.pt \
        --image_path path/to/image.jpg \
        --question "What is in this image?"
"""

import argparse
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import FrameworkConfig, HSREEConfig
from src.training.pipeline import SelfEvolvingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Self-Evolving VLM Inference")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained SR-DPO model")
    parser.add_argument("--gate_path", type=str, required=True,
                        help="Path to the trained confidence head")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--question", type=str, required=True,
                        help="Question about the image")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                        help="Confidence threshold for early exit")
    parser.add_argument("--max_reasoning_steps", type=int, default=5,
                        help="Maximum reasoning depth")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional path to save result as JSON")
    args = parser.parse_args()
    
    # Create config
    config = FrameworkConfig(
        hsr_ee=HSREEConfig(
            confidence_threshold=args.confidence_threshold,
            max_reasoning_steps=args.max_reasoning_steps,
        ),
    )
    
    # Initialize pipeline
    pipeline = SelfEvolvingPipeline(config)
    
    # Run inference
    result = pipeline.inference(
        image=args.image_path,
        question=args.question,
        model_path=args.model_path,
        gate_path=args.gate_path,
    )
    
    # Display result
    print("\n" + "=" * 60)
    print("Self-Evolving VLM Inference Result")
    print("=" * 60)
    print(f"Question: {args.question}")
    print(f"Perception: {result['perception']}")
    print(f"Reasoning Steps: {result['num_steps']}")
    print(f"Exit Type: {result['exit_type']}")
    print(f"Confidence Scores: {result['confidence_scores']}")
    
    if "reasoning_history" in result:
        print("\nReasoning Trace:")
        for i, step in enumerate(result["reasoning_history"]):
            print(f"  Step {i+1}: {step}")
    
    print(f"\nFinal Answer: {result['answer']}")
    print("=" * 60)
    
    # Save to JSON if requested
    if args.output_json:
        # Remove non-serializable items
        save_result = {k: v for k, v in result.items() if k != "hidden_states"}
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(save_result, f, indent=2, ensure_ascii=False)
        print(f"Result saved to {args.output_json}")


if __name__ == "__main__":
    main()
