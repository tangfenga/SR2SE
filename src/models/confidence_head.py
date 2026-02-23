"""
Confidence Head g_phi for HSR-EE Early Exit Mechanism.

From the paper (Section 3.2):
    "We introduce a confidence head g_phi that predicts whether the current 
    history H_k suffices. The head g_phi is a lightweight Multi Layer Perceptron 
    (MLP) that projects the hidden state of the last token of z_k to a scalar score:
        Score_k = sigma(g_phi(H_k))
    If Score_k > tau (a learned threshold), the model terminates the reasoning 
    loop and generates the final answer a."

The training of g_phi uses binary cross-entropy:
    L_gate = -E_k[y_gate * log(p_k) + (1-y_gate) * log(1-p_k)]
where y_gate=1 if current perception yields correct answer (Exit), else 0 (Continue).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ConfidenceHead(nn.Module):
    """
    Lightweight MLP confidence head g_phi for early exit decisions.
    
    Takes the hidden state of the last token of reasoning step z_k
    and projects it to a scalar confidence score via sigmoid.
    
    Architecture: Linear -> ReLU -> Dropout -> Linear -> Sigmoid
    """
    
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Dimension of the VLM's hidden states.
            intermediate_dim: Dimension of the MLP's intermediate layer.
            dropout: Dropout probability.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, 1),
        )
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence score from the hidden state.
        
        Args:
            hidden_state: Hidden state of the last token of z_k.
                Shape: (batch_size, hidden_dim) or (hidden_dim,)
        
        Returns:
            Confidence score in [0, 1]. Shape: (batch_size,) or scalar.
        """
        logit = self.mlp(hidden_state).squeeze(-1)
        score = torch.sigmoid(logit)
        return score
    
    def get_logit(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Return raw logit before sigmoid (useful for gradient analysis).
        
        Args:
            hidden_state: Shape (batch_size, hidden_dim)
        
        Returns:
            Raw logit. Shape: (batch_size,)
        """
        return self.mlp(hidden_state).squeeze(-1)


class GateTrainer:
    """
    Training module for the confidence head g_phi.
    
    The gate is trained as a binary classifier:
    - y_gate = 1 (Exit): current perception yields correct answer
    - y_gate = 0 (Continue): need more reasoning
    
    Loss: Binary cross-entropy
    Additionally uses REINFORCE to adapt the threshold tau.
    """
    
    def __init__(
        self,
        confidence_head: ConfidenceHead,
        learning_rate: float = 1e-3,
        threshold_init: float = 0.7,
        threshold_lr: float = 1e-3,
        length_penalty_lambda: float = 0.1,
        adaptive_threshold: bool = True,
    ):
        self.confidence_head = confidence_head
        self.length_penalty_lambda = length_penalty_lambda
        self.adaptive_threshold = adaptive_threshold
        
        # Learnable threshold tau via REINFORCE
        self.log_tau = nn.Parameter(
            torch.tensor(self._sigmoid_inv(threshold_init)),
            requires_grad=adaptive_threshold,
        )
        
        # Optimizer for gate parameters
        gate_params = list(confidence_head.parameters())
        if adaptive_threshold:
            gate_params.append(self.log_tau)
        self.optimizer = torch.optim.Adam(gate_params, lr=learning_rate)
        self.threshold_lr = threshold_lr
    
    @property
    def tau(self) -> float:
        """Current threshold value."""
        return torch.sigmoid(self.log_tau).item()
    
    @staticmethod
    def _sigmoid_inv(x: float) -> float:
        """Inverse sigmoid for initialization."""
        x = max(min(x, 0.999), 0.001)
        return torch.log(torch.tensor(x / (1 - x))).item()
    
    def compute_gate_loss(
        self,
        confidence_scores: torch.Tensor,
        gate_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute binary cross-entropy loss for the gating module.
        
        L_gate = -E_k[y_gate * log(p_k) + (1-y_gate) * log(1-p_k)]
        
        Args:
            confidence_scores: p_k = sigma(g_phi(H_k)). Shape: (batch_size,)
            gate_labels: y_gate in {0, 1}. Shape: (batch_size,)
        
        Returns:
            Scalar loss.
        """
        loss = F.binary_cross_entropy(
            confidence_scores.clamp(1e-7, 1 - 1e-7),
            gate_labels.float(),
            reduction="mean",
        )
        return loss
    
    def compute_hsr_reward(
        self,
        r_visual: torch.Tensor,
        r_ans: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute HSR-EE reward with length penalty.
        
        R_HSR = r_visual + r_ans - lambda * k
        
        Args:
            r_visual: Visual grounding reward. Shape: (batch_size,)
            r_ans: Answer correctness I(a == a_gt). Shape: (batch_size,)
            k: Number of reasoning steps. Shape: (batch_size,)
        
        Returns:
            HSR reward. Shape: (batch_size,)
        """
        return r_visual + r_ans - self.length_penalty_lambda * k.float()
    
    def update_threshold_reinforce(
        self,
        rewards: torch.Tensor,
        exit_decisions: torch.Tensor,
    ):
        """
        Update threshold tau via REINFORCE.
        
        The threshold affects the exit decisions. We treat the exit decision
        as a policy: P(exit|H_k) = I(Score_k > tau). Since tau is 
        differentiable via the log_tau parametrization, we use REINFORCE
        to optimize it.
        
        Args:
            rewards: R_HSR for each sample. Shape: (batch_size,)
            exit_decisions: Whether model exited at each step. Shape: (batch_size,)
        """
        if not self.adaptive_threshold:
            return
        
        # Baseline: running mean of rewards
        baseline = rewards.mean().detach()
        advantage = rewards - baseline
        
        # REINFORCE: gradient of tau w.r.t. expected reward
        # Higher tau -> fewer exits -> more steps -> lower reward if unnecessary
        tau = torch.sigmoid(self.log_tau)
        
        # Loss: when exiting was beneficial (high reward), lower tau to exit earlier
        # when continuing was beneficial (high reward given no exit), raise tau
        reinforce_loss = -(advantage * exit_decisions.float()).mean() * tau
        
        reinforce_loss.backward(retain_graph=True)
    
    def train_step(
        self,
        hidden_states: torch.Tensor,
        gate_labels: torch.Tensor,
        r_visual: Optional[torch.Tensor] = None,
        r_ans: Optional[torch.Tensor] = None,
        num_steps: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Execute one training step for the confidence head.
        
        Args:
            hidden_states: Hidden states at exit points. Shape: (batch_size, hidden_dim)
            gate_labels: Binary labels {0, 1}. Shape: (batch_size,)
            r_visual: Visual rewards (optional, for REINFORCE).
            r_ans: Answer rewards (optional, for REINFORCE).
            num_steps: Number of steps taken (optional, for REINFORCE).
        
        Returns:
            Dictionary with loss values and metrics.
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        confidence_scores = self.confidence_head(hidden_states)
        
        # Gate loss (binary CE)
        gate_loss = self.compute_gate_loss(confidence_scores, gate_labels)
        
        metrics = {"gate_loss": gate_loss.item(), "tau": self.tau}
        
        # Compute HSR rewards if provided
        if r_visual is not None and r_ans is not None and num_steps is not None:
            hsr_rewards = self.compute_hsr_reward(r_visual, r_ans, num_steps)
            exit_decisions = (confidence_scores > self.tau).float()
            
            # Add REINFORCE loss for threshold
            if self.adaptive_threshold:
                self.update_threshold_reinforce(hsr_rewards, exit_decisions)
            
            metrics["hsr_reward_mean"] = hsr_rewards.mean().item()
        
        gate_loss.backward()
        self.optimizer.step()
        
        metrics["confidence_mean"] = confidence_scores.mean().item()
        return metrics
