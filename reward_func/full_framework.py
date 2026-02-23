"""
Complete Self-Evolving Reward Functions.

Implements all reward and loss computations from the paper:
1. HSR-EE Reward: R_HSR = r_visual + r_ans - lambda * k
2. MDP Step Reward (Bellman formulation)
3. Gate Loss (Binary CE for g_phi)
4. SR-DPO Loss with gradient analysis
5. Utility functions for the self-evolving framework
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SelfEvolvingRewards:
    """
    Unified reward computation module for the Self-Evolving Framework.
    
    Computes:
    - R_HSR: Hierarchical Self-Rewarding with length penalty
    - MDP step rewards for the Bellman formulation
    - Gate BCE loss
    - SR-DPO loss with metrics
    """
    
    def __init__(self, penalty_lambda: float = 0.1, dpo_beta: float = 0.1):
        """
        Args:
            penalty_lambda: Length penalty coefficient lambda (Eq. 12).
            dpo_beta: DPO regularization parameter beta (Section 3.5).
        """
        self.penalty_lambda = penalty_lambda
        self.dpo_beta = dpo_beta
    
    # =====================================================
    # HSR-EE Rewards (Section 3.2)
    # =====================================================
    
    def hsr_ee_reward(self, r_visual: float, r_ans: float, k: int) -> float:
        """
        R_HSR = r_visual + r_ans - lambda * k  (Eq. 12)
        """
        return r_visual + r_ans - (self.penalty_lambda * k)
    
    def hsr_ee_reward_batch(
        self, r_visual: torch.Tensor, r_ans: torch.Tensor, k: torch.Tensor,
    ) -> torch.Tensor:
        """Batch version of hsr_ee_reward."""
        return r_visual + r_ans - self.penalty_lambda * k.float()
    
    # =====================================================
    # MDP Formulation (Section 3.2.3)
    # =====================================================
    
    def mdp_step_reward(
        self, action: str, k: int,
        r_visual: Optional[float] = None, r_ans: Optional[float] = None,
    ) -> float:
        """
        MDP step reward (Eq. 13):
            r(H_k, Exit)     = r_visual + r_ans - lambda*k
            r(H_k, Continue) = -lambda
        """
        if action == "Exit":
            if r_visual is None or r_ans is None:
                raise ValueError("Exit action requires r_visual and r_ans.")
            return r_visual + r_ans - (self.penalty_lambda * k)
        elif action == "Continue":
            return -self.penalty_lambda
        else:
            raise ValueError(f"Invalid action: {action}.")
    
    def bellman_q_value(
        self, action: str, k: int,
        r_visual: float, r_ans: float,
        expected_future_value: float = 0.0,
    ) -> float:
        """Q*(H_k, a_k) under Bellman optimality (Eq. 14-15)."""
        if action == "Exit":
            return r_visual + r_ans - self.penalty_lambda * k
        else:
            return -self.penalty_lambda + expected_future_value
    
    def myopic_utility(self, p_k: float, k: int) -> float:
        """Myopic terminal utility (Eq. 17): U(H_k) = p_k - lambda*k."""
        return p_k - self.penalty_lambda * k
    
    def should_exit(
        self, p_k: float, expected_p_next: float, k: int,
    ) -> bool:
        """Optimal stopping decision (Eq. 18)."""
        return (p_k - self.penalty_lambda * k) >= (
            expected_p_next - self.penalty_lambda * (k + 1)
        )
    
    # =====================================================
    # Gate Loss (Section 3.2.1)
    # =====================================================
    
    def gate_loss(
        self, confidence_scores: torch.Tensor, gate_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Binary CE loss for g_phi (Eq. 11)."""
        return F.binary_cross_entropy(
            confidence_scores.clamp(1e-7, 1 - 1e-7),
            gate_labels.float(),
            reduction="mean",
        )
    
    # =====================================================
    # SR-DPO Loss (Section 3.5)
    # =====================================================
    
    def sr_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        label_smoothing: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        SR-DPO loss (Eq. 20-21):
            f = beta * (log(pi/pi_ref)(y_w|x) - log(pi/pi_ref)(y_l|x))
            L = -E[log sigma(f)]
        """
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps
        
        logits = self.dpo_beta * (chosen_logratios - rejected_logratios)
        
        if label_smoothing > 0:
            losses = (
                -F.logsigmoid(logits) * (1 - label_smoothing)
                - F.logsigmoid(-logits) * label_smoothing
            )
        else:
            losses = -F.logsigmoid(logits)
        
        loss = losses.mean()
        
        with torch.no_grad():
            chosen_rewards = self.dpo_beta * chosen_logratios
            rejected_rewards = self.dpo_beta * rejected_logratios
            reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
            gradient_weight = (1 - torch.sigmoid(logits)).mean()
        
        metrics = {
            "loss": loss.detach(),
            "chosen_rewards": chosen_rewards.mean().detach(),
            "rejected_rewards": rejected_rewards.mean().detach(),
            "reward_accuracy": reward_accuracy,
            "reward_margin": (chosen_rewards - rejected_rewards).mean().detach(),
            "gradient_weight": gradient_weight,
        }
        
        return loss, metrics
    
    # =====================================================
    # Gradient Analysis (Section 3.5.1)
    # =====================================================
    
    @staticmethod
    def dpo_gradient_magnitude(delta: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
        """|dl/dDelta| = beta * (1 - sigma(beta*Delta))  (Eq. 26)"""
        return beta * (1 - torch.sigmoid(beta * delta))
    
    # =====================================================
    # Regret Bound (Proposition 1)
    # =====================================================
    
    @staticmethod
    def compute_regret_bound(
        marginal_gains: np.ndarray, lambda_val: float, epsilon: float,
    ) -> Dict[str, float]:
        """
        Regret bound: R <= epsilon * W_epsilon,
        W_epsilon = |{k : |Delta_k - lambda| <= epsilon}|.
        """
        K = len(marginal_gains)
        J = np.zeros(K + 1)
        for k in range(K):
            J[k + 1] = J[k] + marginal_gains[k] - lambda_val
        
        k_star = int(np.argmax(J))
        k_hat = K
        for k in range(K):
            if marginal_gains[k] <= lambda_val:
                k_hat = k + 1
                break
        
        W_epsilon = int(np.sum(np.abs(marginal_gains - lambda_val) <= epsilon))
        
        return {
            "regret_bound": epsilon * W_epsilon,
            "W_epsilon": W_epsilon,
            "k_star": k_star,
            "k_hat": k_hat,
            "J_star": float(J[k_star]),
            "J_hat": float(J[min(k_hat, K)]),
            "actual_regret": float(J[k_star] - J[min(k_hat, K)]),
        }