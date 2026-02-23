"""
Script: Train HSR-EE confidence head (gate module g_phi).

Usage:
    python scripts/train_hsr_ee.py \
        --model_path ./output/sft_ckpt \
        --dataset_path path/to/dataset.jsonl \
        --output_dir ./output/hsr_ee_ckpt \
        --confidence_threshold 0.7 \
        --max_reasoning_steps 5 \
        --length_penalty_lambda 0.1
"""

import argparse
import json
import logging
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import HSREEConfig
from src.models.vlm_wrapper import VLMWrapper
from src.models.confidence_head import ConfidenceHead
from src.hsr_ee.inference import HSREETrainer

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
    return data


def main():
    parser = argparse.ArgumentParser(description="Train HSR-EE confidence head")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the SFT checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the training dataset")
    parser.add_argument("--output_dir", type=str, default="./output/hsr_ee_ckpt",
                        help="Output directory for gate checkpoint")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                        help="Initial confidence threshold tau")
    parser.add_argument("--max_reasoning_steps", type=int, default=5,
                        help="Maximum reasoning depth K")
    parser.add_argument("--length_penalty_lambda", type=float, default=0.1,
                        help="Length penalty lambda")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs for gate")
    parser.add_argument("--threshold_lr", type=float, default=1e-3,
                        help="Learning rate for gate + threshold")
    parser.add_argument("--device_map", type=str, default="auto")
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    logger.info(f"Loaded {len(dataset)} training samples")
    
    # Initialize VLM
    vlm = VLMWrapper(
        model_name_or_path=args.model_path,
        device_map=args.device_map,
    )
    
    # Configure HSR-EE
    config = HSREEConfig(
        confidence_threshold=args.confidence_threshold,
        max_reasoning_steps=args.max_reasoning_steps,
        length_penalty_lambda=args.length_penalty_lambda,
        adaptive_threshold=True,
        threshold_lr=args.threshold_lr,
    )
    
    # Initialize confidence head
    confidence_head = ConfidenceHead(
        hidden_dim=vlm.hidden_size,
        intermediate_dim=config.confidence_head_hidden_dim,
    )
    if torch.cuda.is_available():
        confidence_head = confidence_head.cuda()
    
    # Initialize trainer
    trainer = HSREETrainer(
        vlm=vlm,
        confidence_head=confidence_head,
        config=config,
    )
    
    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_count = 0
        
        for i, sample in enumerate(dataset):
            try:
                trajectory = trainer.collect_trajectory(
                    image=sample.get("image", sample.get("image_path", "")),
                    question=sample["question"],
                    ground_truth=sample["answer"],
                )
                metrics = trainer.train_step(trajectory)
                
                if not metrics.get("skipped", False):
                    epoch_loss += metrics.get("gate_loss", 0)
                    epoch_count += 1
                
                if (i + 1) % 50 == 0:
                    avg_loss = epoch_loss / max(epoch_count, 1)
                    logger.info(
                        f"Epoch {epoch+1}, Step {i+1}/{len(dataset)}: "
                        f"gate_loss={avg_loss:.4f}, "
                        f"tau={metrics.get('tau', config.confidence_threshold):.4f}"
                    )
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        # Save epoch checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"gate_epoch_{epoch+1}.pt")
        torch.save(confidence_head.state_dict(), checkpoint_path)
        logger.info(
            f"Epoch {epoch+1}/{args.num_epochs}: avg_loss={epoch_loss / max(epoch_count, 1):.4f}"
        )
    
    # Save final checkpoint
    final_path = os.path.join(args.output_dir, "confidence_head.pt")
    torch.save(confidence_head.state_dict(), final_path)
    logger.info(f"Training complete. Gate saved to {final_path}")


if __name__ == "__main__":
    main()
