"""
IPR: Iterative Perception Refinement.

From the paper (Section 3.4, Algorithm 3):
    IPR operates as an offline module for constructing the preference dataset
    D_pref. It uses ground truth labels (a_gt) to validate perceptions, 
    enabling the synthesis of high quality Winner-Loser pairs from failure
    trajectories of the model.

    Detection:
        Valid(c_0) = I(pi_text(Q, c_0) == a_gt)
    
    Refinement:
        When Valid(c_0) = 0, prompt the model to re-examine the image.
        c_refined = Refine(I, Q, c_0)
        If Valid(c_refined) = 1: y_w = c_refined, y_l = c_0
    
    Pair Construction:
        (y_w, y_l) pairs where y_w is the refined perception (high quality)
        and y_l is the original perception (hallucinated/incomplete).

    Max iterations: N_max = 2 (optimal window from ablation study).
"""

import json
import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm

from src.models.vlm_wrapper import VLMWrapper
from src.config import IPRConfig

logger = logging.getLogger(__name__)


@dataclass
class PreferencePair:
    """A single preference pair for SR-DPO training."""
    image_path: str
    question: str
    ground_truth: str
    winner: str          # y_w: refined (correct) perception
    loser: str           # y_l: original (hallucinated) perception
    refinement_iteration: int  # Which iteration succeeded (1 or 2)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IPRStatistics:
    """Statistics from an IPR data generation run."""
    total_samples: int = 0
    initially_correct: int = 0
    refined_success: int = 0
    refined_failed: int = 0
    pairs_generated: int = 0
    refinement_iter_1_success: int = 0
    refinement_iter_2_success: int = 0
    
    @property
    def success_rate(self) -> float:
        attempted = self.total_samples - self.initially_correct
        if attempted == 0:
            return 0.0
        return self.refined_success / attempted
    
    @property
    def pair_yield(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.pairs_generated / self.total_samples
    
    def __str__(self) -> str:
        return (
            f"IPR Statistics:\n"
            f"  Total samples: {self.total_samples}\n"
            f"  Initially correct: {self.initially_correct}\n"
            f"  Refinement success: {self.refined_success}\n"
            f"  Refinement failed: {self.refined_failed}\n"
            f"  Pairs generated: {self.pairs_generated}\n"
            f"  Iteration 1 success: {self.refinement_iter_1_success}\n"
            f"  Iteration 2 success: {self.refinement_iter_2_success}\n"
            f"  Success rate: {self.success_rate:.2%}\n"
            f"  Pair yield: {self.pair_yield:.2%}\n"
        )


class IPRDataEngine:
    """
    Iterative Perception Refinement Data Engine.
    
    Implements Algorithm 3 from the paper:
    1. Generate initial perception c_0 = pi_perc(I, Q)
    2. Validate: Valid(c_0) = I(pi_text(Q, c_0) == a_gt)
    3. If valid: skip (no preference pair needed)
    4. If invalid: refine up to N_max times
       - c_refined = Refine(I, Q, c_prev)
       - If Valid(c_refined): return (y_w=c_refined, y_l=c_0)
    5. If all refinements fail: discard sample
    """
    
    def __init__(
        self,
        vlm: VLMWrapper,
        config: IPRConfig,
    ):
        self.vlm = vlm
        self.config = config
        self.stats = IPRStatistics()
    
    def validate_perception(
        self,
        question: str,
        perception: str,
        ground_truth: str,
    ) -> bool:
        """
        Valid(c) = I(pi_text(Q, c) == a_gt)
        
        Check if the perception is sufficient to answer the question correctly.
        
        Args:
            question: The question Q.
            perception: The visual perception c.
            ground_truth: The ground truth answer a_gt.
        
        Returns:
            True if perception yields correct answer.
        """
        # Use text-only policy to answer from perception
        answer = self.vlm.answer_from_perception(question, perception)
        return self._check_answer(answer, ground_truth)
    
    def process_single_sample(
        self,
        image: Any,
        question: str,
        ground_truth: str,
        image_path: str = "",
    ) -> Optional[PreferencePair]:
        """
        Process a single sample through the IPR pipeline (Algorithm 3).
        
        Args:
            image: Input image.
            question: Query Q.
            ground_truth: Ground truth answer a_gt.
            image_path: Path to the image file.
        
        Returns:
            PreferencePair if refinement succeeds, None otherwise.
        """
        self.stats.total_samples += 1
        
        # Step 1: Generate initial perception
        c_0 = self.vlm.perceive(image, question)
        
        # Step 2: Validate initial perception
        if self.validate_perception(question, c_0, ground_truth):
            self.stats.initially_correct += 1
            logger.debug(f"Sample already correct, skipping refinement.")
            return None
        
        # Step 3: Iterative refinement
        c_current = c_0
        for iteration in range(1, self.config.max_refinement_iterations + 1):
            # Generate refined perception
            c_refined = self.vlm.refine_perception(
                image=image,
                question=question,
                previous_perception=c_current,
            )
            
            # Validate refined perception
            if self.validate_perception(question, c_refined, ground_truth):
                # Success! Create preference pair
                self.stats.refined_success += 1
                self.stats.pairs_generated += 1
                
                if iteration == 1:
                    self.stats.refinement_iter_1_success += 1
                elif iteration == 2:
                    self.stats.refinement_iter_2_success += 1
                
                logger.debug(
                    f"Refinement succeeded at iteration {iteration}."
                )
                
                return PreferencePair(
                    image_path=image_path,
                    question=question,
                    ground_truth=ground_truth,
                    winner=c_refined,    # y_w: corrected perception
                    loser=c_0,           # y_l: original hallucinated perception
                    refinement_iteration=iteration,
                )
            
            # Update for next iteration
            c_current = c_refined
        
        # All refinement attempts failed
        self.stats.refined_failed += 1
        logger.debug(
            f"Refinement failed after {self.config.max_refinement_iterations} iterations."
        )
        return None
    
    def generate_preference_dataset(
        self,
        dataset: List[Dict[str, Any]],
        output_path: str,
        max_samples: Optional[int] = None,
    ) -> Tuple[List[PreferencePair], IPRStatistics]:
        """
        Generate the full preference dataset D_pref from training data.
        
        Args:
            dataset: List of training examples, each with keys:
                     'image' (path or PIL), 'question', 'answer'.
            output_path: Path to save the preference dataset.
            max_samples: Maximum number of samples to process.
        
        Returns:
            Tuple of (list of preference pairs, statistics).
        """
        self.stats = IPRStatistics()  # Reset statistics
        preference_pairs: List[PreferencePair] = []
        
        n_samples = min(len(dataset), max_samples or len(dataset))
        logger.info(f"Starting IPR data generation on {n_samples} samples...")
        
        for idx in tqdm(range(n_samples), desc="IPR Data Generation"):
            sample = dataset[idx]
            image = sample.get("image", sample.get("image_path", ""))
            question = sample["question"]
            ground_truth = sample["answer"]
            image_path = sample.get("image_path", str(idx))
            
            pair = self.process_single_sample(
                image=image,
                question=question,
                ground_truth=ground_truth,
                image_path=image_path,
            )
            
            if pair is not None:
                preference_pairs.append(pair)
        
        # Save preference dataset
        self._save_dataset(preference_pairs, output_path)
        
        logger.info(f"IPR data generation complete.\n{self.stats}")
        return preference_pairs, self.stats
    
    def _save_dataset(
        self,
        pairs: List[PreferencePair],
        output_path: str,
    ):
        """Save preference pairs to JSONL file."""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(pairs)} preference pairs to {output_path}")
    
    @staticmethod
    def load_dataset(path: str) -> List[PreferencePair]:
        """Load preference pairs from JSONL file."""
        pairs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                pairs.append(PreferencePair(**data))
        return pairs
    
    @staticmethod
    def _check_answer(predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth."""
        pred = predicted.strip().lower()
        gt = ground_truth.strip().lower()
        
        if pred == gt:
            return True
        if gt in pred:
            return True
        if len(gt) == 1 and pred.startswith(gt):
            return True
        
        return False
