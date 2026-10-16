"""Configuration for models, datasets, inference, and prompts.

Exposes:
  * Typed dataclass schemas (ModelConfig, DataConfig, InferenceConfig) and
    enums (ModelFamily, QuestionType, DatasetName) — see ``schemas``.
  * Model registry mapping short keys to HuggingFace IDs — see ``registry``.
  * Prompt templates for sampling and verbalized methods — see ``prompts``.
  * Default hyperparameters (sampling N, seed, etc.) — see ``defaults``.
"""
from .schemas import (
    ModelFamily, QuestionType, DatasetName,
    ModelConfig, DataConfig, InferenceConfig,
)
from .registry import (
    MODELS, MODEL_SHORT_KEYS, MODEL_NUM_GPUS, MODEL_NUM_GPUS_4BIT,
)
from .prompts import (
    VERBALIZED_VARIANTS, VERBALIZED_PROMPTS, LINGUISTIC_MAP,
    BASE_SAMPLING_PROMPT, COT_SAMPLING_PROMPT,
)
from .defaults import (
    DATASETS, QUESTION_TYPES, NUM_BINS, SAMPLING_N, SAMPLING_TEMP, SEED,
    JUDGE_MODEL_ID,
)

__all__ = [
    # schemas
    "ModelFamily", "QuestionType", "DatasetName",
    "ModelConfig", "DataConfig", "InferenceConfig",
    # registry
    "MODELS", "MODEL_SHORT_KEYS", "MODEL_NUM_GPUS", "MODEL_NUM_GPUS_4BIT",
    # prompts
    "VERBALIZED_VARIANTS", "VERBALIZED_PROMPTS", "LINGUISTIC_MAP",
    "BASE_SAMPLING_PROMPT", "COT_SAMPLING_PROMPT",
    # defaults
    "DATASETS", "QUESTION_TYPES", "NUM_BINS", "SAMPLING_N", "SAMPLING_TEMP",
    "SEED", "JUDGE_MODEL_ID",
]
