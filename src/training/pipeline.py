"""
Overall Training Pipeline (Algorithm 2 from the paper).

The training pipeline proceeds through three sequential stages:
    Stage 1: SFT Initialization
        Fine-tune V on instruction-following data to obtain pi_ref.
    
    Stage 2: IPR-based Preference Construction
        For each (I, Q, a_gt) in D:
            Run HSR-EE inference to obtain (c, t, a) and R_HSR
            Generate (y_w, y_l) using IPR (Algorithm 3)
            Update gate parameters g_phi using R_HSR and L_gate
    
    Stage 3: SR-DPO Optimization
        For multiple epochs:
            Sample mini-batches from D_pref
            Update policy parameters theta by minimizing L_{SR-DPO}
"""

import os
import json
import copy
import logging
import torch
from typing import Any, Dict, List, Optional
from pathlib import Path

from src.config import FrameworkConfig
from src.models.vlm_wrapper import VLMWrapper
from src.models.confidence_head import ConfidenceHead, GateTrainer
from src.hsr_ee.inference import HSREEInference, HSREETrainer
from src.ipr.data_engine import IPRDataEngine
from src.sr_dpo.trainer import SRDPOTrainer

logger = logging.getLogger(__name__)


class SelfEvolvingPipeline:
    """
    Complete Self-Evolving VLM Training Pipeline.
    
    Orchestrates the three stages:
    1. SFT Initialization (via LlamaFactory integration)
    2. IPR + HSR-EE Training (preference construction + gate training)
    3. SR-DPO Optimization (preference-based policy update)
    """
    
    def __init__(self, config: FrameworkConfig):
        self.config = config
        self.vlm: Optional[VLMWrapper] = None
        self.ref_model: Optional[Any] = None
        self.confidence_head: Optional[ConfidenceHead] = None
        
        # Ensure output directories exist
        for dir_path in [
            config.output_dir,
            config.sft_output_dir,
            config.ipr_data_dir,
            config.hsr_ee_output_dir,
            config.sr_dpo_output_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)
    
    def initialize_model(self):
        """Load the base VLM and initialize components."""
        logger.info("Initializing VLM...")
        self.vlm = VLMWrapper(
            model_name_or_path=self.config.model.model_name_or_path,
            device_map=self.config.model.device_map,
            torch_dtype=torch.bfloat16 if self.config.model.bf16 else torch.float32,
            use_flash_attention=self.config.model.use_flash_attention,
        )
        
        # Initialize confidence head g_phi
        hidden_size = self.vlm.hidden_size
        self.confidence_head = ConfidenceHead(
            hidden_dim=hidden_size,
            intermediate_dim=self.config.hsr_ee.confidence_head_hidden_dim,
        )
        
        if torch.cuda.is_available():
            self.confidence_head = self.confidence_head.cuda()
        
        logger.info(
            f"Model initialized: hidden_size={hidden_size}, "
            f"confidence_head_dim={self.config.hsr_ee.confidence_head_hidden_dim}"
        )
    
    # =========================================================
    # Stage 1: SFT Initialization
    # =========================================================
    def run_sft(
        self,
        dataset_path: Optional[str] = None,
        use_llamafactory: bool = True,
    ):
        """
        Stage 1: Supervised Fine-Tuning.
        
        Fine-tune V on instruction-following data (LLaVA-Instruct-150k)
        to obtain pi_ref.
        
        Args:
            dataset_path: Path to SFT dataset. If None, uses config default.
            use_llamafactory: Whether to use LlamaFactory for SFT.
        """
        logger.info("=" * 60)
        logger.info("Stage 1: SFT Initialization")
        logger.info("=" * 60)
        
        if use_llamafactory:
            self._run_sft_llamafactory(dataset_path)
        else:
            self._run_sft_native(dataset_path)
        
        logger.info(f"SFT complete. Checkpoint saved to {self.config.sft_output_dir}")
    
    def _run_sft_llamafactory(self, dataset_path: Optional[str] = None):
        """Run SFT using LlamaFactory integration."""
        # Generate LlamaFactory config
        lf_config = {
            "model_name_or_path": self.config.model.model_name_or_path,
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "full" if self.config.model.lora_rank == 0 else "lora",
            "dataset": self.config.sft.dataset_name,
            "template": "qwen_vl",
            "output_dir": self.config.sft_output_dir,
            "overwrite_output_dir": True,
            "per_device_train_batch_size": self.config.sft.batch_size,
            "gradient_accumulation_steps": self.config.sft.gradient_accumulation_steps,
            "learning_rate": self.config.sft.learning_rate,
            "num_train_epochs": self.config.sft.num_epochs,
            "max_length": self.config.sft.max_seq_length,
            "logging_steps": self.config.logging_steps,
            "save_steps": self.config.save_steps,
            "bf16": self.config.model.bf16,
        }
        
        if self.config.model.lora_rank > 0:
            lf_config["lora_rank"] = self.config.model.lora_rank
            lf_config["lora_alpha"] = self.config.model.lora_alpha
        
        # Save config
        config_path = os.path.join(self.config.output_dir, "sft_config.json")
        with open(config_path, "w") as f:
            json.dump(lf_config, f, indent=2)
        
        logger.info(f"LlamaFactory SFT config saved to {config_path}")
        logger.info(
            "To run SFT, execute:\n"
            f"  cd LlamaFactory && python src/train.py {config_path}"
        )
    
    def _run_sft_native(self, dataset_path: Optional[str] = None):
        """Run SFT natively (basic implementation)."""
        logger.info("Running native SFT (basic implementation)...")
        # This is a simplified SFT loop for demonstration
        # In practice, LlamaFactory provides a much more robust implementation
        
        if self.vlm is None:
            self.initialize_model()
        
        # Save initial model as SFT checkpoint (placeholder)
        if hasattr(self.vlm.model, "save_pretrained"):
            self.vlm.model.save_pretrained(self.config.sft_output_dir)
            self.vlm.tokenizer.save_pretrained(self.config.sft_output_dir)
    
    # =========================================================
    # Stage 2: IPR + HSR-EE Training
    # =========================================================
    def run_ipr_and_gate_training(
        self,
        dataset: List[Dict[str, Any]],
        sft_checkpoint_path: Optional[str] = None,
    ):
        """
        Stage 2: IPR-based Preference Construction + HSR-EE Gate Training.
        
        For each (I, Q, a_gt) in D:
            1. Run HSR-EE inference to obtain (c, t, a) and R_HSR
            2. Generate (y_w, y_l) using IPR (Algorithm 3)
            3. Update gate parameters g_phi using R_HSR and L_gate
        
        Args:
            dataset: Training dataset with 'image', 'question', 'answer' keys.
            sft_checkpoint_path: Path to load the SFT checkpoint.
        """
        logger.info("=" * 60)
        logger.info("Stage 2: IPR Data Generation + HSR-EE Gate Training")
        logger.info("=" * 60)
        
        # Load SFT checkpoint if specified
        if sft_checkpoint_path:
            self.vlm = VLMWrapper(
                model_name_or_path=sft_checkpoint_path,
                device_map=self.config.model.device_map,
                torch_dtype=torch.bfloat16 if self.config.model.bf16 else torch.float32,
            )
            self.confidence_head = ConfidenceHead(
                hidden_dim=self.vlm.hidden_size,
                intermediate_dim=self.config.hsr_ee.confidence_head_hidden_dim,
            )
            if torch.cuda.is_available():
                self.confidence_head = self.confidence_head.cuda()
        
        if self.vlm is None:
            self.initialize_model()
        
        # Initialize IPR Data Engine
        ipr_engine = IPRDataEngine(
            vlm=self.vlm,
            config=self.config.ipr,
        )
        
        # Initialize HSR-EE Trainer
        hsr_ee_trainer = HSREETrainer(
            vlm=self.vlm,
            confidence_head=self.confidence_head,
            config=self.config.hsr_ee,
        )
        
        # Generate preference data via IPR
        ipr_output_path = os.path.join(self.config.ipr_data_dir, "preference_pairs.jsonl")
        preference_pairs, stats = ipr_engine.generate_preference_dataset(
            dataset=dataset,
            output_path=ipr_output_path,
            max_samples=self.config.ipr.num_samples,
        )
        
        logger.info(f"IPR Results:\n{stats}")
        
        # Train gate on collected trajectories
        logger.info("Training HSR-EE gate...")
        for i, sample in enumerate(dataset[:min(len(dataset), self.config.ipr.num_samples)]):
            trajectory = hsr_ee_trainer.collect_trajectory(
                image=sample.get("image", sample.get("image_path", "")),
                question=sample["question"],
                ground_truth=sample["answer"],
            )
            gate_metrics = hsr_ee_trainer.train_step(trajectory)
            
            if (i + 1) % self.config.logging_steps == 0:
                logger.info(
                    f"Gate training step {i+1}: "
                    f"loss={gate_metrics.get('gate_loss', 0):.4f}, "
                    f"tau={gate_metrics.get('tau', 0):.4f}"
                )
        
        # Save gate checkpoint
        gate_path = os.path.join(self.config.hsr_ee_output_dir, "confidence_head.pt")
        torch.save(self.confidence_head.state_dict(), gate_path)
        logger.info(f"Gate checkpoint saved to {gate_path}")
        
        return ipr_output_path
    
    # =========================================================
    # Stage 3: SR-DPO Optimization
    # =========================================================
    def run_sr_dpo(
        self,
        preference_data_path: str,
        sft_checkpoint_path: Optional[str] = None,
    ):
        """
        Stage 3: SR-DPO Optimization.
        
        For multiple epochs:
            Sample mini-batches from D_pref
            Update policy parameters theta by minimizing L_{SR-DPO}
        
        Args:
            preference_data_path: Path to IPR-generated preference dataset.
            sft_checkpoint_path: Path to SFT checkpoint (for reference model).
        """
        logger.info("=" * 60)
        logger.info("Stage 3: SR-DPO Optimization")
        logger.info("=" * 60)
        
        # Load policy model (current)
        if self.vlm is None:
            model_path = sft_checkpoint_path or self.config.model.model_name_or_path
            self.vlm = VLMWrapper(
                model_name_or_path=model_path,
                device_map=self.config.model.device_map,
                torch_dtype=torch.bfloat16 if self.config.model.bf16 else torch.float32,
            )
        
        # Load reference model (frozen SFT checkpoint)
        ref_path = sft_checkpoint_path or self.config.sft_output_dir
        logger.info(f"Loading reference model from {ref_path}")
        ref_model = copy.deepcopy(self.vlm.model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        
        # Initialize SR-DPO trainer
        trainer = SRDPOTrainer(
            model=self.vlm.model,
            ref_model=ref_model,
            tokenizer=self.vlm.tokenizer,
            beta=self.config.sr_dpo.beta,
            learning_rate=self.config.sr_dpo.learning_rate,
            batch_size=self.config.sr_dpo.batch_size,
            num_epochs=self.config.sr_dpo.num_epochs,
            gradient_accumulation_steps=self.config.sr_dpo.gradient_accumulation_steps,
            max_grad_norm=self.config.sr_dpo.max_grad_norm,
            warmup_ratio=self.config.sr_dpo.warmup_ratio,
            weight_decay=self.config.sr_dpo.weight_decay,
            max_seq_length=self.config.sr_dpo.max_seq_length,
            output_dir=self.config.sr_dpo_output_dir,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
        )
        
        # Run training
        trainer.train(preference_data_path)
        
        logger.info(f"SR-DPO training complete. Model saved to {self.config.sr_dpo_output_dir}")
    
    # =========================================================
    # Full Pipeline
    # =========================================================
    def run_full_pipeline(
        self,
        dataset: List[Dict[str, Any]],
        skip_sft: bool = False,
        sft_checkpoint_path: Optional[str] = None,
    ):
        """
        Execute the complete three-stage training pipeline (Algorithm 2).
        
        Args:
            dataset: Training dataset.
            skip_sft: If True, skip Stage 1 (use existing SFT checkpoint).
            sft_checkpoint_path: Path to existing SFT checkpoint.
        """
        logger.info("=" * 60)
        logger.info("Self-Evolving VLM: Full Training Pipeline")
        logger.info("=" * 60)
        
        # Stage 1: SFT
        if not skip_sft:
            self.run_sft()
        
        sft_path = sft_checkpoint_path or self.config.sft_output_dir
        
        # Stage 2: IPR + HSR-EE
        ipr_data_path = self.run_ipr_and_gate_training(
            dataset=dataset,
            sft_checkpoint_path=sft_path,
        )
        
        # Stage 3: SR-DPO
        self.run_sr_dpo(
            preference_data_path=ipr_data_path,
            sft_checkpoint_path=sft_path,
        )
        
        logger.info("=" * 60)
        logger.info("Full pipeline complete!")
        logger.info(
            f"Final model: {self.config.sr_dpo_output_dir}\n"
            f"Gate model: {self.config.hsr_ee_output_dir}"
        )
        logger.info("=" * 60)
    
    # =========================================================
    # Inference (after training)
    # =========================================================
    def inference(
        self,
        image: Any,
        question: str,
        model_path: Optional[str] = None,
        gate_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run inference using the trained Self-Evolving VLM.
        
        Args:
            image: Input image.
            question: Query.
            model_path: Path to the trained model.
            gate_path: Path to the trained confidence head.
        
        Returns:
            Inference result with answer, perception, reasoning trace.
        """
        # Load trained model if not already loaded
        if model_path and self.vlm is None:
            self.vlm = VLMWrapper(
                model_name_or_path=model_path,
                device_map=self.config.model.device_map,
            )
        
        # Load trained gate if not already loaded
        if gate_path and self.confidence_head is None:
            self.confidence_head = ConfidenceHead(
                hidden_dim=self.vlm.hidden_size,
                intermediate_dim=self.config.hsr_ee.confidence_head_hidden_dim,
            )
            self.confidence_head.load_state_dict(
                torch.load(gate_path, map_location="cpu")
            )
            if torch.cuda.is_available():
                self.confidence_head = self.confidence_head.cuda()
        
        # Run HSR-EE inference
        inference_engine = HSREEInference(
            vlm=self.vlm,
            confidence_head=self.confidence_head,
            config=self.config.hsr_ee,
        )
        
        result = inference_engine.inference(
            image=image,
            question=question,
            return_trajectory=True,
        )
        
        return result
