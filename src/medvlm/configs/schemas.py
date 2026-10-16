"""Configuration classes for the Medical VQA framework."""
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class ModelFamily(str, Enum):
    """Supported VLM model families."""

    QWEN_VL = "qwen_vl"
    INTERNVL = "internvl"
    LLAVA = "llava"
    LLAVA_NEXT = "llava_next"
    GEMMA = "gemma"


class QuestionType(str, Enum):
    """VQA question type filter (closed yes/no, open-ended, or all)."""

    ALL = "all"
    CLOSED = "closed"
    OPEN = "open"


class DatasetName(str, Enum):
    """Registry of available medical VQA benchmark datasets."""

    VQA_RAD = "vqa_rad"
    SLAKE = "slake"
    VQA_MED_2019 = "vqa_med_2019"
    VQA_MED_2020 = "vqa_med_2020"
    VQA_MED_2021 = "vqa_med_2021"


@dataclass
class ModelConfig:
    """Configuration for loading a VLM.

    Specifies the HuggingFace model ID, quantization settings (4-bit/8-bit),
    attention backend, and dtype. Model family is auto-detected from model_id
    if not provided.
    """

    model_id: str
    model_family: Optional[ModelFamily] = None
    use_4bit: bool = False
    use_8bit: bool = False
    use_flash_attention: bool = True
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True

    def __post_init__(self):
        if self.model_family is None:
            self.model_family = self._detect_family()

    def _detect_family(self) -> ModelFamily:
        model_id_lower = self.model_id.lower()
        if (
            ("qwen" in model_id_lower and "vl" in model_id_lower)
            or "lingshu" in model_id_lower
            or "medvlm-r1" in model_id_lower
            or "medvlm_r1" in model_id_lower
        ):
            return ModelFamily.QWEN_VL
        elif "internvl" in model_id_lower:
            return ModelFamily.INTERNVL
        elif "llava" in model_id_lower:
            if "next" in model_id_lower or "1.6" in model_id_lower:
                return ModelFamily.LLAVA_NEXT
            return ModelFamily.LLAVA
        elif "gemma" in model_id_lower:
            return ModelFamily.GEMMA
        else:
            raise ValueError(f"Cannot detect model family for: {self.model_id}")


@dataclass
class DataConfig:
    """Configuration for dataset loading, filtering, and subsampling."""

    dataset_name: DatasetName
    question_type: QuestionType = QuestionType.ALL
    split: str = "train"
    subsample_size: Optional[int] = None
    seed: int = 42
    data_path: Optional[str] = None
    image_dir: Optional[str] = None


@dataclass
class InferenceConfig:
    """Configuration for VQA inference (sampling parameters, batching, adapters)."""

    adapter_path: Optional[str] = None
    num_samples: int = 1
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    batch_size: int = 1
