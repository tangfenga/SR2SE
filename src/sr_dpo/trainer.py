"""
SR-DPO: Self-Rewarding Direct Preference Optimization.

From the paper (Section 3.5):
    L_{SR-DPO} = -E_{(x,y_w,y_l)~D} [log sigma(
        beta * log(pi_theta(y_w|x) / pi_ref(y_w|x)) 
        - beta * log(pi_theta(y_l|x) / pi_ref(y_l|x))
    )]
    
    where x = (I, Q), y_w is the preferred (refined) perception,
    y_l is the rejected (hallucinated) perception, and pi_ref is 
    the SFT checkpoint.

Key properties:
    - Gradient: nabla L = -E[(1-sigma(f_theta)) * beta * (nabla log pi(y_w|x) - nabla log pi(y_l|x))]
    - Hard negatives from IPR provide concentrated gradient signal
    - Variance reduction vs Random-DPO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import os
import copy
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class DPOBatch:
    """A batch of preference data for SR-DPO training."""
    # Input context (image path + question concatenated for text representation)
    input_ids_chosen: torch.Tensor      # Tokenized (context + y_w)
    attention_mask_chosen: torch.Tensor
    input_ids_rejected: torch.Tensor    # Tokenized (context + y_l)
    attention_mask_rejected: torch.Tensor
    labels_chosen: torch.Tensor         # Labels for y_w tokens only
    labels_rejected: torch.Tensor       # Labels for y_l tokens only


class PreferenceDataset(Dataset):
    """
    Dataset for SR-DPO training from IPR-generated preference pairs.
    
    Each sample contains:
        - context: Image representation + question
        - chosen (y_w): Refined, correct perception
        - rejected (y_l): Original, hallucinated perception
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
    
    def _load_data(self, path: str) -> List[dict]:
        """Load preference pairs from JSONL file."""
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        logger.info(f"Loaded {len(data)} preference pairs from {path}")
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # Build context: question (ideally image features would be prepended)
        context = f"Question: {item['question']}\nVisual Perception:"
        
        # Chosen (y_w) = refined perception
        chosen_text = context + " " + item["winner"]
        # Rejected (y_l) = original perception
        rejected_text = context + " " + item["loser"]
        
        # Tokenize
        chosen_tokens = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        rejected_tokens = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Compute context length for label masking
        context_tokens = self.tokenizer(
            context,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        context_len = context_tokens["input_ids"].shape[1]
        
        # Create labels: -100 for context tokens (don't compute loss on them)
        chosen_labels = chosen_tokens["input_ids"].clone().squeeze(0)
        chosen_labels[:context_len] = -100
        
        rejected_labels = rejected_tokens["input_ids"].clone().squeeze(0)
        rejected_labels[:context_len] = -100
        
        return {
            "input_ids_chosen": chosen_tokens["input_ids"].squeeze(0),
            "attention_mask_chosen": chosen_tokens["attention_mask"].squeeze(0),
            "input_ids_rejected": rejected_tokens["input_ids"].squeeze(0),
            "attention_mask_rejected": rejected_tokens["attention_mask"].squeeze(0),
            "labels_chosen": chosen_labels,
            "labels_rejected": rejected_labels,
        }


class SRDPOLoss(nn.Module):
    """
    Self-Rewarding Direct Preference Optimization Loss.
    
    L_{SR-DPO} = -E[log sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x) 
                                      - log pi(y_l|x)/pi_ref(y_l|x)))]
    
    Gradient:
        nabla L = -E[(1-sigma(f_theta)) * beta * 
                     (nabla log pi(y_w|x) - nabla log pi(y_l|x))]
    """
    
    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute SR-DPO loss.
        
        Args:
            policy_chosen_logps: log pi_theta(y_w|x). Shape: (batch_size,)
            policy_rejected_logps: log pi_theta(y_l|x). Shape: (batch_size,)
            ref_chosen_logps: log pi_ref(y_w|x). Shape: (batch_size,)
            ref_rejected_logps: log pi_ref(y_l|x). Shape: (batch_size,)
        
        Returns:
            Tuple of (loss, metrics_dict).
        """
        # f_theta = beta * (log(pi_theta(y_w)/pi_ref(y_w)) - log(pi_theta(y_l)/pi_ref(y_l)))
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps
        
        logits = self.beta * (chosen_logratios - rejected_logratios)
        
        # L = -E[log sigma(f_theta)]
        if self.label_smoothing > 0:
            losses = (
                -F.logsigmoid(logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-logits) * self.label_smoothing
            )
        else:
            losses = -F.logsigmoid(logits)
        
        loss = losses.mean()
        
        # Compute metrics for monitoring
        with torch.no_grad():
            # Reward difference (implicit reward)
            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
            reward_margin = (chosen_rewards - rejected_rewards).mean()
            
            # Gradient magnitude proxy: (1 - sigma(f_theta))
            gradient_weight = (1 - torch.sigmoid(logits)).mean()
        
        metrics = {
            "loss": loss.detach(),
            "chosen_rewards": chosen_rewards.mean().detach(),
            "rejected_rewards": rejected_rewards.mean().detach(),
            "reward_accuracy": reward_accuracy,
            "reward_margin": reward_margin,
            "gradient_weight": gradient_weight,
            "logits_mean": logits.mean().detach(),
        }
        
        return loss, metrics


def compute_sequence_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the sum of log-probabilities for the target tokens.
    
    Args:
        model: The language model.
        input_ids: Input token IDs. Shape: (batch_size, seq_len)
        attention_mask: Attention mask. Shape: (batch_size, seq_len)
        labels: Labels with -100 for context tokens. Shape: (batch_size, seq_len)
    
    Returns:
        Sum of log-probs for each sequence. Shape: (batch_size,)
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits
    
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    
    # Compute per-token log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather log probs at label positions
    # Replace -100 with 0 for gathering, then mask them out
    gather_labels = shift_labels.clone()
    gather_labels[gather_labels == -100] = 0
    
    per_token_logps = log_probs.gather(
        dim=-1, index=gather_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # Mask out context tokens (where label was -100)
    mask = (shift_labels != -100).float()
    per_token_logps = per_token_logps * mask
    
    # Sum per sequence
    return per_token_logps.sum(dim=-1)


class SRDPOTrainer:
    """
    Full SR-DPO Training loop.
    
    Implements Stage 3 of the overall training pipeline:
    1. Load preference dataset D_pref from IPR
    2. Initialize reference model pi_ref (frozen SFT checkpoint)
    3. For multiple epochs:
        - Sample mini-batches from D_pref
        - Compute SR-DPO loss
        - Update policy parameters theta
    """
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        tokenizer: Any,
        beta: float = 0.1,
        learning_rate: float = 5e-7,
        batch_size: int = 64,
        num_epochs: int = 3,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_seq_length: int = 2048,
        output_dir: str = "./output/sr_dpo_ckpt",
        save_steps: int = 500,
        logging_steps: int = 10,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.max_seq_length = max_seq_length
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        
        # Freeze reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Loss function
        self.loss_fn = SRDPOLoss(beta=beta)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler (will be set during training)
        self.scheduler = None
        self.warmup_ratio = warmup_ratio
        self.global_step = 0
    
    def _create_scheduler(self, num_training_steps: int):
        """Create a linear warmup + cosine decay schedule."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
        
        warmup_steps = int(num_training_steps * self.warmup_ratio)
        
        warmup = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps - warmup_steps,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    
    def train(self, data_path: str):
        """
        Execute the full SR-DPO training loop.
        
        Args:
            data_path: Path to the preference dataset (JSONL from IPR).
        """
        # Create dataset and dataloader
        dataset = PreferenceDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=self.max_seq_length,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        num_training_steps = (
            len(dataloader) * self.num_epochs 
            // self.gradient_accumulation_steps
        )
        self._create_scheduler(num_training_steps)
        
        logger.info(
            f"Starting SR-DPO training:\n"
            f"  Samples: {len(dataset)}\n"
            f"  Epochs: {self.num_epochs}\n"
            f"  Batch size: {self.batch_size}\n"
            f"  Total steps: {num_training_steps}\n"
        )
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.model.train()
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_metrics = {}
            
            progress_bar = tqdm(
                dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}"
            )
            
            for step, batch in enumerate(progress_bar):
                metrics = self._train_step(batch)
                
                epoch_loss += metrics["loss"]
                for k, v in metrics.items():
                    if k not in epoch_metrics:
                        epoch_metrics[k] = 0.0
                    epoch_metrics[k] += v
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    progress_bar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        reward_acc=f"{metrics.get('reward_accuracy', 0):.3f}",
                    )
                
                # Saving
                if self.global_step % self.save_steps == 0 and self.global_step > 0:
                    self._save_checkpoint(
                        os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
                    )
            
            # Log epoch summary
            n_batches = len(dataloader)
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Loss: {epoch_loss / n_batches:.4f} - "
                f"Reward Accuracy: {epoch_metrics.get('reward_accuracy', 0) / n_batches:.3f}"
            )
        
        # Save final model
        self._save_checkpoint(os.path.join(self.output_dir, "final"))
        logger.info(f"SR-DPO training complete. Model saved to {self.output_dir}")
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute one training step.
        
        Args:
            batch: Dictionary with tokenized inputs.
        
        Returns:
            Metrics dictionary.
        """
        device = next(self.model.parameters()).device
        
        # Move batch to device
        input_ids_chosen = batch["input_ids_chosen"].to(device)
        attention_mask_chosen = batch["attention_mask_chosen"].to(device)
        input_ids_rejected = batch["input_ids_rejected"].to(device)
        attention_mask_rejected = batch["attention_mask_rejected"].to(device)
        labels_chosen = batch["labels_chosen"].to(device)
        labels_rejected = batch["labels_rejected"].to(device)
        
        # Compute policy log-probs
        policy_chosen_logps = compute_sequence_log_probs(
            self.model, input_ids_chosen, attention_mask_chosen, labels_chosen
        )
        policy_rejected_logps = compute_sequence_log_probs(
            self.model, input_ids_rejected, attention_mask_rejected, labels_rejected
        )
        
        # Compute reference log-probs (no gradient)
        with torch.no_grad():
            ref_chosen_logps = compute_sequence_log_probs(
                self.ref_model, input_ids_chosen, attention_mask_chosen, labels_chosen
            )
            ref_rejected_logps = compute_sequence_log_probs(
                self.ref_model, input_ids_rejected, attention_mask_rejected, labels_rejected
            )
        
        # Compute SR-DPO loss
        loss, metrics = self.loss_fn(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
        )
        
        # Gradient accumulation
        scaled_loss = loss / self.gradient_accumulation_steps
        scaled_loss.backward()
        
        self.global_step += 1
        
        if self.global_step % self.gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}
    
    def _save_checkpoint(self, path: str):
        """Save model checkpoint."""
        os.makedirs(path, exist_ok=True)
        
        # Save model
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(path)
        else:
            torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        
        # Save tokenizer
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(path)
        
        # Save optimizer state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
        }, os.path.join(path, "training_state.pt"))
        
        logger.info(f"Checkpoint saved to {path}")
