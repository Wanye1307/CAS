"""
Inference module for Medical VQA models.

Supports:
- Base model inference
- SFT fine-tuned model inference (via LoRA adapters)
- Single sample and batch inference
- Sampling-based inference for calibration
"""

import os
import json
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from tqdm import tqdm

import torch
from datasets import Dataset

from ..configs import ModelConfig, DataConfig, InferenceConfig, ModelFamily
from ..data import get_dataset
from ..models import load_model


@dataclass
class VQAPrediction:
    """Single VQA prediction result."""
    question: str
    ground_truth: str
    predictions: List[str]
    answer_type: str

    # Additional metadata
    image_id: Optional[str] = None
    question_id: Optional[str] = None


class VQAInference:
    """Unified inference class for Medical VQA models."""

    def __init__(
        self,
        model_config: ModelConfig,
        inference_config: InferenceConfig,
    ):
        """Initialize inference engine.

        Args:
            model_config: Model configuration
            inference_config: Inference settings
        """
        self.model_config = model_config
        self.inference_config = inference_config

        self.model = None
        self.processor = None

    def load_model(self) -> None:
        """Load model with optional adapter."""
        print(f"Loading model: {self.model_config.model_id}")

        self.model, self.processor = load_model(
            self.model_config,
            adapter_path=self.inference_config.adapter_path,
        )

        self.model.eval()

        if self.inference_config.adapter_path:
            print(f"Loaded adapter: {self.inference_config.adapter_path}")
        else:
            print("Using base model (no adapter)")

    def _format_prompt(self, image: Any, question: str) -> Dict:
        """Format input for the model based on model family."""

        if self.model_config.model_family == ModelFamily.QWEN_VL:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

        elif self.model_config.model_family == ModelFamily.INTERNVL:
            # InternVL3-hf uses same format as Qwen-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                },
            ]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.processor(
                images=image,
                text=text,
                return_tensors="pt"
            )

        elif self.model_config.model_family in [ModelFamily.LLAVA, ModelFamily.LLAVA_NEXT]:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                },
            ]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt"
            )

        else:
            raise ValueError(f"Unsupported model family: {self.model_config.model_family}")

        return inputs

    def predict_single(
        self,
        image: Any,
        question: str,
        num_samples: Optional[int] = None,
    ) -> List[str]:
        """Generate predictions for a single question.

        Args:
            image: PIL Image
            question: Question text
            num_samples: Number of samples to generate (overrides config)

        Returns:
            List of prediction strings
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        n_samples = num_samples or self.inference_config.num_samples

        # Format and move to device
        inputs = self._format_prompt(image, question)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=self.inference_config.do_sample,
                num_return_sequences=n_samples,
                max_new_tokens=self.inference_config.max_new_tokens,
                temperature=self.inference_config.temperature,
                top_p=self.inference_config.top_p,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        # Decode (trim prompt)
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[:, prompt_len:]
        predictions = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        return predictions

    def predict_dataset(
        self,
        dataset: Dataset,
        num_samples: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[VQAPrediction]:
        """Generate predictions for an entire dataset.

        Args:
            dataset: HuggingFace dataset with image, question, answer columns
            num_samples: Samples per question for calibration
            show_progress: Whether to show progress bar

        Returns:
            List of VQAPrediction objects
        """
        results = []

        iterator = tqdm(dataset, desc="Inference") if show_progress else dataset

        for example in iterator:
            predictions = self.predict_single(
                example["image"],
                example["question"],
                num_samples=num_samples,
            )

            result = VQAPrediction(
                question=example["question"],
                ground_truth=example["answer"],
                predictions=predictions,
                answer_type=example.get("answer_type", "unknown"),
                image_id=example.get("image_id"),
                question_id=example.get("question_id"),
            )
            results.append(result)

        return results

    def save_results(
        self,
        results: List[VQAPrediction],
        output_path: str,
    ) -> None:
        """Save inference results to JSON file.

        Args:
            results: List of VQAPrediction objects
            output_path: Path to save JSON file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert to serializable format
        data = []
        for r in results:
            data.append({
                "question": r.question,
                "ground_truth": r.ground_truth,
                "predictions": r.predictions,
                "answer_type": r.answer_type,
                "image_id": r.image_id,
                "question_id": r.question_id,
            })

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to: {output_path}")


def run_inference(
    model_config: ModelConfig,
    data_config: DataConfig,
    inference_config: InferenceConfig,
    output_path: str,
) -> List[VQAPrediction]:
    """Run inference on a dataset.

    Args:
        model_config: Model configuration
        data_config: Dataset configuration
        inference_config: Inference settings
        output_path: Path to save results

    Returns:
        List of predictions
    """
    # Load dataset
    print(f"Loading dataset: {data_config.dataset_name.value}")
    dataset_wrapper = get_dataset(data_config)
    dataset = dataset_wrapper.load()

    print(f"Dataset size: {len(dataset)}")

    # Setup inference
    inference = VQAInference(model_config, inference_config)
    inference.load_model()

    # Run inference
    results = inference.predict_dataset(
        dataset,
        num_samples=inference_config.num_samples,
    )

    # Save results
    inference.save_results(results, output_path)

    return results
