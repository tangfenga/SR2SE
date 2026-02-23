"""
HSR-EE: Hierarchical Self-Rewarding with Early Exit.

From the paper (Section 3.2, Algorithm 1):
    The inference process is modeled as a dynamic graph. At each step k,
    the model maintains a history state H_k and generates a sub-goal z_k.
    A confidence head g_phi predicts whether H_k suffices:
        Score_k = sigma(g_phi(H_k))
    If Score_k > tau, early exit; otherwise continue reasoning.

    R_HSR = r_visual + r_ans - lambda * k

This module implements:
    1. HSR-EE Inference (Algorithm 1)
    2. HSR-EE Training (gate training + REINFORCE for threshold)
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
import logging

from src.models.confidence_head import ConfidenceHead, GateTrainer
from src.models.vlm_wrapper import VLMWrapper
from src.config import HSREEConfig

logger = logging.getLogger(__name__)


class HSREEInference:
    """
    HSR-EE Inference Engine (Algorithm 1 from the paper).
    
    Implements the dynamic reasoning loop with early exit:
    1. c = pi_perc(I, Q)           # Perception via prompting
    2. For k = 1..K:
        z_k = pi_reason(c, Q, H)   # Generate next reasoning step
        Score_k = sigma(g_phi(H))   # Compute confidence
        If Score_k > tau: break     # Early exit
    3. a = GenerateAnswer(H)        # Final answer
    """
    
    def __init__(
        self,
        vlm: VLMWrapper,
        confidence_head: ConfidenceHead,
        config: HSREEConfig,
    ):
        self.vlm = vlm
        self.confidence_head = confidence_head
        self.config = config
        self._tau = config.confidence_threshold
    
    @property
    def tau(self) -> float:
        """Current early exit threshold."""
        return self._tau
    
    @tau.setter
    def tau(self, value: float):
        self._tau = max(0.0, min(1.0, value))
    
    def inference(
        self,
        image: Any,
        question: str,
        return_trajectory: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute HSR-EE inference (Algorithm 1).
        
        Args:
            image: Input image (PIL Image or path).
            question: Query Q.
            return_trajectory: If True, return full reasoning trajectory.
        
        Returns:
            Dictionary containing:
                - 'answer': Final answer a
                - 'perception': Visual perception c
                - 'num_steps': Number of reasoning steps k
                - 'confidence_scores': List of confidence scores at each step
                - 'reasoning_history': List of reasoning steps (if return_trajectory)
                - 'exit_type': 'early' or 'max_steps'
        """
        # Step 1: Generate visual perception
        perception = self.vlm.perceive(image, question)
        
        # Step 2: Hierarchical reasoning with early exit
        history: List[str] = []
        confidence_scores: List[float] = []
        hidden_states: List[torch.Tensor] = []
        exit_type = "max_steps"
        
        for k in range(self.config.max_reasoning_steps):
            # Generate next reasoning step
            z_k, hidden_state = self.vlm.reason_step(
                perception=perception,
                question=question,
                history=history,
                image=image,
            )
            history.append(z_k)
            
            # Compute confidence score
            if hidden_state is not None:
                score = self.confidence_head(hidden_state).item()
            else:
                # Fallback: use the text representation
                history_text = f"Perception: {perception}\n" + "\n".join(
                    [f"Step {i+1}: {h}" for i, h in enumerate(history)]
                )
                hidden = self.vlm.get_hidden_state_for_text(history_text)
                score = self.confidence_head(hidden).item()
            
            confidence_scores.append(score)
            if hidden_state is not None:
                hidden_states.append(hidden_state)
            
            logger.debug(
                f"Step {k+1}: confidence={score:.4f}, tau={self.tau:.4f}"
            )
            
            # Early exit check
            if score > self.tau:
                exit_type = "early"
                logger.info(
                    f"Early exit at step {k+1}/{self.config.max_reasoning_steps} "
                    f"(score={score:.4f} > tau={self.tau:.4f})"
                )
                break
        
        # Step 3: Generate final answer
        answer = self.vlm.generate_final_answer(
            question=question,
            perception=perception,
            reasoning_history=history,
        )
        
        result = {
            "answer": answer,
            "perception": perception,
            "num_steps": len(history),
            "confidence_scores": confidence_scores,
            "exit_type": exit_type,
        }
        
        if return_trajectory:
            result["reasoning_history"] = history
            result["hidden_states"] = hidden_states
        
        return result
    
    def batch_inference(
        self,
        images: List[Any],
        questions: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Execute HSR-EE inference on a batch of samples.
        
        Args:
            images: List of input images.
            questions: List of questions.
        
        Returns:
            List of inference results.
        """
        results = []
        for image, question in zip(images, questions):
            result = self.inference(image, question, return_trajectory=True)
            results.append(result)
        return results


class HSREETrainer:
    """
    HSR-EE Training Module.
    
    Combines:
    1. Gate training (binary CE loss for g_phi)
    2. REINFORCE for adaptive threshold tau
    3. HSR reward computation
    """
    
    def __init__(
        self,
        vlm: VLMWrapper,
        confidence_head: ConfidenceHead,
        config: HSREEConfig,
    ):
        self.vlm = vlm
        self.config = config
        
        # Initialize gate trainer
        self.gate_trainer = GateTrainer(
            confidence_head=confidence_head,
            learning_rate=config.threshold_lr,
            threshold_init=config.confidence_threshold,
            threshold_lr=config.threshold_lr,
            length_penalty_lambda=config.length_penalty_lambda,
            adaptive_threshold=config.adaptive_threshold,
        )
        
        # Inference engine (uses same components)
        self.inference_engine = HSREEInference(
            vlm=vlm,
            confidence_head=confidence_head,
            config=config,
        )
    
    def collect_trajectory(
        self,
        image: Any,
        question: str,
        ground_truth: str,
    ) -> Dict[str, Any]:
        """
        Collect a training trajectory from a single sample.
        
        For gate training, we need:
        - Hidden states at each reasoning step
        - Pseudo-labels: y_gate=1 if perception yields correct answer
        - HSR rewards
        
        Args:
            image: Input image.
            question: Query.
            ground_truth: Ground truth answer a_gt.
        
        Returns:
            Trajectory data for gate training.
        """
        # Run inference with trajectory recording
        result = self.inference_engine.inference(
            image, question, return_trajectory=True
        )
        
        perception = result["perception"]
        history = result.get("reasoning_history", [])
        hidden_states = result.get("hidden_states", [])
        answer = result["answer"]
        num_steps = result["num_steps"]
        
        # Compute rewards
        # r_ans = I(a == a_gt)
        r_ans = float(self._check_answer(answer, ground_truth))
        
        # r_visual: check if perception alone can yield correct answer
        answer_from_perc = self.vlm.answer_from_perception(question, perception)
        r_visual = float(self._check_answer(answer_from_perc, ground_truth))
        
        # Gate pseudo-labels: y_gate=1 if perception is sufficient (exit now)
        # For each step k, check if the history up to k would yield correct answer
        gate_labels = []
        for k in range(num_steps):
            partial_answer = self.vlm.generate_final_answer(
                question, perception, history[:k+1]
            )
            is_correct = self._check_answer(partial_answer, ground_truth)
            gate_labels.append(float(is_correct))
        
        return {
            "perception": perception,
            "reasoning_history": history,
            "hidden_states": hidden_states,
            "answer": answer,
            "ground_truth": ground_truth,
            "num_steps": num_steps,
            "r_ans": r_ans,
            "r_visual": r_visual,
            "gate_labels": gate_labels,
            "confidence_scores": result["confidence_scores"],
            "exit_type": result["exit_type"],
        }
    
    def train_step(self, trajectory: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform one gate training step from a collected trajectory.
        
        Args:
            trajectory: Trajectory data from collect_trajectory.
        
        Returns:
            Training metrics.
        """
        hidden_states = trajectory["hidden_states"]
        gate_labels = trajectory["gate_labels"]
        r_visual = trajectory["r_visual"]
        r_ans = trajectory["r_ans"]
        num_steps = trajectory["num_steps"]
        
        if not hidden_states:
            return {"gate_loss": 0.0, "skipped": True}
        
        # Stack tensors
        hidden_tensor = torch.stack(hidden_states)
        labels_tensor = torch.tensor(gate_labels, device=hidden_tensor.device)
        r_visual_tensor = torch.tensor(
            [r_visual] * len(hidden_states), device=hidden_tensor.device
        )
        r_ans_tensor = torch.tensor(
            [r_ans] * len(hidden_states), device=hidden_tensor.device
        )
        steps_tensor = torch.arange(
            1, len(hidden_states) + 1, device=hidden_tensor.device
        ).float()
        
        metrics = self.gate_trainer.train_step(
            hidden_states=hidden_tensor,
            gate_labels=labels_tensor,
            r_visual=r_visual_tensor,
            r_ans=r_ans_tensor,
            num_steps=steps_tensor,
        )
        
        return metrics
    
    @staticmethod
    def _check_answer(predicted: str, ground_truth: str) -> bool:
        """
        Check if predicted answer matches ground truth.
        Simple string matching with normalization.
        """
        pred = predicted.strip().lower()
        gt = ground_truth.strip().lower()
        
        # Exact match
        if pred == gt:
            return True
        
        # Check if ground truth is contained in prediction
        if gt in pred:
            return True
        
        # Check first word / letter match for multiple choice
        if len(gt) == 1 and pred.startswith(gt):
            return True
        
        return False
