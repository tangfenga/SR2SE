"""
VLM Wrapper with Sub-Policy Decomposition.

From the paper (Section 3.1):
    "We parameterize the vision language model as V(theta). Prompting extracts 
    the sub-policies pi_perc, pi_reason, and pi_text from V."
    
    - pi_perc(I, Q) -> c (visual perception)
    - pi_reason(c, Q, H_{k-1}) -> z_k (reasoning step)
    - pi_text(Q, c) -> a (answer from perception)

This module provides a unified interface wrapping Qwen-VL-Chat or similar VLMs
and exposing the sub-policy decomposition for use in HSR-EE, IPR, and SR-DPO.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


# ---- Prompt Templates ---- #

PERCEPTION_PROMPT = (
    "You are a visual perception assistant. Given the image and the question below, "
    "describe what you see in the image that is relevant to answering the question. "
    "Focus on visual details, spatial relationships, colors, objects, and text visible "
    "in the image.\n\n"
    "Question: {question}\n\n"
    "Visual Perception:"
)

REASONING_PROMPT = (
    "Based on the visual perception and previous reasoning steps, continue reasoning "
    "about the question. Provide the next logical step.\n\n"
    "Question: {question}\n"
    "Visual Perception: {perception}\n"
    "Previous Reasoning: {history}\n\n"
    "Next Reasoning Step:"
)

ANSWER_FROM_PERCEPTION_PROMPT = (
    "Based only on the following visual description, answer the question. "
    "Give a concise answer.\n\n"
    "Question: {question}\n"
    "Visual Description: {perception}\n\n"
    "Answer:"
)

FULL_ANSWER_PROMPT = (
    "Based on the visual perception and reasoning, provide the final answer.\n\n"
    "Question: {question}\n"
    "Visual Perception: {perception}\n"
    "Reasoning: {reasoning}\n\n"
    "Final Answer:"
)

REFINEMENT_PROMPT = (
    "The description '{perception}' was insufficient to answer the question correctly. "
    "Please re-examine the image. Focus on spatial relationships and small details "
    "that were missed. Provide a refined description.\n\n"
    "Question: {question}\n\n"
    "Refined Visual Perception:"
)


class VLMWrapper(nn.Module):
    """
    Wrapper around a Vision-Language Model that exposes sub-policies
    for the Self-Evolving Framework.
    
    Sub-policies:
        - pi_perc: Perception policy (image + query -> visual description)
        - pi_reason: Reasoning policy (perception + query + history -> next step)
        - pi_text: Text-only answer policy (query + perception -> answer)
    """
    
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen-VL-Chat",
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        use_flash_attention: bool = True,
        max_new_tokens: int = 512,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.use_flash_attention = use_flash_attention
    
    @property
    def model(self):
        """Lazy-load the model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        """Lazy-load the tokenizer."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer
    
    @property
    def hidden_size(self) -> int:
        """Return the hidden dimension of the underlying model."""
        if self._model is not None:
            return self._model.config.hidden_size
        # Default for Qwen-VL-Chat 7B
        return 4096
    
    def _load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model from {self.model_name_or_path}")
        
        attn_impl = "flash_attention_2" if self.use_flash_attention else "eager"
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
        logger.info(f"Model loaded: {type(self._model).__name__}")
    
    def _generate(
        self,
        prompt: str,
        image: Optional[Any] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        do_sample: bool = True,
        return_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate text from the VLM given a prompt and optional image.
        
        Args:
            prompt: The text prompt.
            image: Image input (PIL Image or path).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling.
            return_hidden_states: Whether to return the last hidden state.
        
        Returns:
            Dictionary with 'text', and optionally 'hidden_state'.
        """
        max_tokens = max_new_tokens or self.max_new_tokens
        
        # Build query for Qwen-VL-Chat format
        if image is not None:
            # For Qwen-VL, images are embedded via special tokens
            if isinstance(image, str):
                query_content = [
                    {"image": image},
                    {"text": prompt},
                ]
            else:
                # PIL Image: save to temp or use direct
                query_content = [{"text": prompt}]
        else:
            query_content = [{"text": prompt}]
        
        # Use the model's chat interface if available
        if hasattr(self.model, "chat"):
            response, _ = self.model.chat(
                self.tokenizer,
                query=prompt if image is None else query_content,
                history=None,
            )
            result = {"text": response}
        else:
            # Standard generation
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    output_hidden_states=return_hidden_states,
                    return_dict_in_generate=True,
                )
            
            generated_ids = outputs.sequences[:, inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            result = {"text": text}
            
            if return_hidden_states and hasattr(outputs, "hidden_states"):
                # Get the last hidden state of the last generated token
                last_hidden = outputs.hidden_states[-1][-1][:, -1, :]
                result["hidden_state"] = last_hidden
        
        return result
    
    def get_hidden_state_for_text(self, text: str) -> torch.Tensor:
        """
        Get the hidden state of the last token for a given text input.
        Used for the confidence head.
        
        Args:
            text: Input text (concatenation of perception + reasoning history).
        
        Returns:
            Hidden state tensor of shape (hidden_dim,).
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
            )
        
        # Last layer, last token hidden state
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        return last_hidden.squeeze(0)
    
    # =====================================================
    # Sub-Policy: pi_perc (Perception)
    # =====================================================
    def perceive(
        self,
        image: Any,
        question: str,
        max_new_tokens: int = 256,
    ) -> str:
        """
        pi_perc(I, Q) -> c
        
        Generate visual perception description for the image given the question.
        
        Args:
            image: Image input (PIL Image or path).
            question: The question to answer.
            max_new_tokens: Max tokens for perception.
        
        Returns:
            Perception text c.
        """
        prompt = PERCEPTION_PROMPT.format(question=question)
        result = self._generate(prompt, image=image, max_new_tokens=max_new_tokens)
        return result["text"]
    
    # =====================================================
    # Sub-Policy: pi_reason (Reasoning)
    # =====================================================
    def reason_step(
        self,
        perception: str,
        question: str,
        history: List[str],
        image: Optional[Any] = None,
        max_new_tokens: int = 256,
    ) -> Tuple[str, Optional[torch.Tensor]]:
        """
        pi_reason(c, Q, H_{k-1}) -> z_k
        
        Generate the next reasoning step given perception, question, and history.
        
        Args:
            perception: Visual perception text c.
            question: The question.
            history: List of previous reasoning steps [z_1, ..., z_{k-1}].
            image: Optional image for multi-modal reasoning.
            max_new_tokens: Max tokens for this reasoning step.
        
        Returns:
            Tuple of (reasoning step text z_k, hidden state for confidence head).
        """
        history_str = " ".join(
            [f"Step {i+1}: {h}" for i, h in enumerate(history)]
        ) if history else "None"
        
        prompt = REASONING_PROMPT.format(
            question=question,
            perception=perception,
            history=history_str,
        )
        result = self._generate(
            prompt,
            image=image,
            max_new_tokens=max_new_tokens,
            return_hidden_states=True,
        )
        
        hidden_state = result.get("hidden_state", None)
        return result["text"], hidden_state
    
    # =====================================================
    # Sub-Policy: pi_text (Answer from Perception)
    # =====================================================
    def answer_from_perception(
        self,
        question: str,
        perception: str,
        max_new_tokens: int = 128,
    ) -> str:
        """
        pi_text(Q, c) -> a
        
        Generate answer from perception text alone (no image).
        Used in IPR validation: Valid(c) = I(pi_text(Q, c) == a_gt)
        
        Args:
            question: The question.
            perception: Visual perception text c.
            max_new_tokens: Max tokens for the answer.
        
        Returns:
            Answer text a.
        """
        prompt = ANSWER_FROM_PERCEPTION_PROMPT.format(
            question=question,
            perception=perception,
        )
        result = self._generate(prompt, max_new_tokens=max_new_tokens)
        return result["text"]
    
    # =====================================================
    # Full Answer Generation (with reasoning history)
    # =====================================================
    def generate_final_answer(
        self,
        question: str,
        perception: str,
        reasoning_history: List[str],
        max_new_tokens: int = 128,
    ) -> str:
        """
        Generate the final answer based on perception and full reasoning chain.
        
        Args:
            question: The question.
            perception: Visual perception text c.
            reasoning_history: List of reasoning steps [z_1, ..., z_K].
            max_new_tokens: Max tokens for the answer.
        
        Returns:
            Final answer text a.
        """
        reasoning_str = " ".join(
            [f"Step {i+1}: {h}" for i, h in enumerate(reasoning_history)]
        ) if reasoning_history else "No additional reasoning needed."
        
        prompt = FULL_ANSWER_PROMPT.format(
            question=question,
            perception=perception,
            reasoning=reasoning_str,
        )
        result = self._generate(prompt, max_new_tokens=max_new_tokens)
        return result["text"]
    
    # =====================================================
    # Refinement Generation (for IPR)
    # =====================================================
    def refine_perception(
        self,
        image: Any,
        question: str,
        previous_perception: str,
        max_new_tokens: int = 256,
    ) -> str:
        """
        Generate a refined perception after a failed validation.
        Used in the IPR module.
        
        Args:
            image: Image input.
            question: The question.
            previous_perception: The failed perception c_0.
            max_new_tokens: Max tokens for refinement.
        
        Returns:
            Refined perception c_refined.
        """
        prompt = REFINEMENT_PROMPT.format(
            perception=previous_perception,
            question=question,
        )
        result = self._generate(prompt, image=image, max_new_tokens=max_new_tokens)
        return result["text"]
    
    # =====================================================
    # Log-Probability Computation (for SR-DPO)
    # =====================================================
    def compute_log_probs(
        self,
        input_text: str,
        target_text: str,
    ) -> torch.Tensor:
        """
        Compute log P(target | input) under the current model.
        Used for SR-DPO loss computation.
        
        Args:
            input_text: The conditioning context (image description + question).
            target_text: The response to score (perception/answer).
        
        Returns:
            Sum of log-probabilities for the target sequence.
        """
        full_text = input_text + target_text
        
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
        ).to(self.model.device)
        
        input_len = len(self.tokenizer(input_text)["input_ids"])
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Shift for next-token prediction
        shift_logits = logits[:, input_len - 1:-1, :]
        shift_labels = inputs["input_ids"][:, input_len:]
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        return token_log_probs.sum(dim=-1)
    
    def forward(self, *args, **kwargs):
        """Forward pass delegates to the underlying model."""
        return self.model(*args, **kwargs)
