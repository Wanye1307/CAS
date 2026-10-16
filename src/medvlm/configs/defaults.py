"""Default constants for datasets, evaluation, and sampling."""
from __future__ import annotations

from typing import List

DATASETS: List[str] = [
    "vqa_rad", "slake", "vqa_med_2019", "vqa_med_2020", "vqa_med_2021",
]
QUESTION_TYPES: List[str] = ["closed", "open"]

NUM_BINS: int = 15
SAMPLING_N: int = 20
SAMPLING_TEMP: float = 0.7
SEED: int = 42

JUDGE_MODEL_ID: str = "Qwen/Qwen3-4B-Instruct-2507"
