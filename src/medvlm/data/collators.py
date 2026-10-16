"""
Data collators for different VLM model families.

Each collator handles the specific input format required by its model family,
including chat template formatting and multimodal input processing.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import torch

from ..configs import ModelFamily


class BaseVQACollator(ABC):
    """Abstract base class for VQA data collators."""

    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    @abstractmethod
    def format_messages(self, sample: Dict) -> List[Dict]:
        """Format a sample into chat messages."""
        pass

    @abstractmethod
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples."""
        pass


class QwenVLCollator(BaseVQACollator):
    """Collator for Qwen-VL models.

    Note: Qwen-VL has special image tokens that must NOT be truncated,
    similar to LLaVA-NeXT. We disable truncation to preserve image tokens.
    """

    def format_messages(self, sample: Dict) -> List[Dict]:
        """Format for Qwen-VL chat template."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": sample["question"]},
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["answer"]}]
            }
        ]

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch for Qwen-VL.

        IMPORTANT: Do NOT truncate - Qwen-VL image tokens will be corrupted.
        """
        texts = []
        images = []

        for example in examples:
            messages = self.format_messages(example)
            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text_prompt)
            images.append(example["image"])

        # Process batch - NO truncation to preserve image tokens
        # Qwen-VL image tokens get corrupted if truncated
        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
            # Do NOT truncate - image tokens must be preserved
        )

        # Create labels (same as input_ids for causal LM)
        batch["labels"] = batch["input_ids"].clone()

        return batch


class InternVLCollator(BaseVQACollator):
    """Collator for InternVL3 models (HuggingFace version).

    InternVL3-hf uses the same chat format as Qwen-VL with AutoProcessor.
    """

    def format_messages(self, sample: Dict) -> List[Dict]:
        """Format for InternVL3 chat template (same as Qwen-VL)."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": sample["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["answer"]},
                ],
            },
        ]

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch for InternVL3-hf."""
        images = []
        texts = []

        for example in examples:
            images.append(example["image"])
            messages = self.format_messages(example)
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)

        # Process images and text together (same as Qwen-VL)
        # NOTE: Do NOT use truncation - it breaks image token alignment
        batch = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=False,  # Important: don't truncate image tokens!
        )

        # Create labels
        labels = batch["input_ids"].clone()
        if hasattr(self.processor, 'tokenizer'):
            pad_token_id = self.processor.tokenizer.pad_token_id
        else:
            pad_token_id = getattr(self.processor, 'pad_token_id', None)

        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        batch["labels"] = labels

        return batch


class LLaVACollator(BaseVQACollator):
    """Collator for LLaVA models (1.5 and earlier).

    Note: LLaVA also has image tokens that shouldn't be truncated.
    """

    def format_messages(self, sample: Dict) -> List[Dict]:
        """Format for LLaVA chat template."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": sample["question"]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["answer"]}
                ]
            }
        ]

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch for LLaVA."""
        texts = []
        images = []

        for example in examples:
            messages = self.format_messages(example)
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
            images.append(example["image"])

        # Process batch - NO truncation to preserve image tokens
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            # Do NOT truncate - image tokens must be preserved
        )

        # Create labels
        labels = batch["input_ids"].clone()
        tokenizer = getattr(self.processor, 'tokenizer', self.processor)
        pad_id = getattr(tokenizer, 'pad_token_id', None)
        if pad_id is not None:
            labels[labels == pad_id] = -100
        batch["labels"] = labels

        return batch


class LLaVANextCollator(BaseVQACollator):
    """Collator for LLaVA-NeXT models (1.6+).

    Note: LLaVA-NeXT has many image tokens that shouldn't be truncated.
    """

    def format_messages(self, sample: Dict) -> List[Dict]:
        """Format for LLaVA-NeXT chat template."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": sample["question"]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["answer"]}
                ]
            }
        ]

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch for LLaVA-NeXT - no truncation to preserve image tokens."""
        texts = []
        images = []

        for example in examples:
            messages = self.format_messages(example)
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
            images.append(example["image"])

        # Process batch - NO truncation for LLaVA-NeXT
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            # Don't truncate - image tokens must be preserved
        )

        # Create labels
        labels = batch["input_ids"].clone()
        tokenizer = getattr(self.processor, 'tokenizer', self.processor)
        pad_id = getattr(tokenizer, 'pad_token_id', None)
        if pad_id is not None:
            labels[labels == pad_id] = -100
        batch["labels"] = labels

        return batch


# Collator Registry
_COLLATOR_REGISTRY: Dict[ModelFamily, type] = {
    ModelFamily.QWEN_VL: QwenVLCollator,
    ModelFamily.INTERNVL: InternVLCollator,
    ModelFamily.LLAVA: LLaVACollator,
    ModelFamily.LLAVA_NEXT: LLaVANextCollator,
}


def get_collator(
    model_family: ModelFamily,
    processor,
    max_length: int = 2048
) -> BaseVQACollator:
    """Factory function to get the appropriate collator."""
    if model_family not in _COLLATOR_REGISTRY:
        raise ValueError(f"Unknown model family: {model_family}. "
                        f"Available: {list(_COLLATOR_REGISTRY.keys())}")

    collator_cls = _COLLATOR_REGISTRY[model_family]
    return collator_cls(processor, max_length)
