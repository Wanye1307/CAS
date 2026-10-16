"""Registry of supported VLMs.

Maps short keys (used across the CLI, experiments, and examples) to HuggingFace
model IDs, plus GPU requirements for full-precision and 4-bit loading.
"""
from __future__ import annotations

import os

from typing import Dict

MODELS: Dict[str, str] = {
    "qwen3vl_2b": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3vl_4b_local": os.path.join(os.environ.get("CAS_MODEL_ROOT", "models"), "Qwen3-VL-4B-Instruct"),
    "qwen3vl_8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3vl_32b": "Qwen/Qwen3-VL-32B-Instruct",
    "internvl3_2b": "OpenGVLab/InternVL3-2B-hf",
    "internvl3_2b_local": os.path.join(os.environ.get("CAS_MODEL_ROOT", "models"), "InternVL3-2B-hf"),
    "internvl3_8b": "OpenGVLab/InternVL3-8B-hf",
    "internvl3_8b_local": os.path.join(os.environ.get("CAS_MODEL_ROOT", "models"), "InternVL3-8B-hf"),
    "internvl3_38b": "OpenGVLab/InternVL3-38B-hf",
    "llava_next_7b": "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava_next_7b_local": os.path.join(os.environ.get("CAS_MODEL_ROOT", "models"), "llava-v1.6-mistral-7b-hf"),
    "llava_next_34b": "llava-hf/llava-v1.6-34b-hf",
    "llava_med": os.path.join(os.environ.get("CAS_MODEL_ROOT", "models"), "llava-med-v1.5-mistral-7b-hf"),
    "medgemma_4b": os.path.join(os.environ.get("CAS_MODEL_ROOT", "models"), "MedGemma-4B"),
    "lingshu_7b": os.path.join(os.environ.get("CAS_MODEL_ROOT", "models"), "Lingshu-7B"),
    "medvlm_r1": os.path.join(os.environ.get("CAS_MODEL_ROOT", "models"), "MedVLM-R1"),
}

MODEL_SHORT_KEYS: Dict[str, str] = {
    "qwen3vl_2b": "qwen2b",
    "qwen3vl_4b_local": "qwen4b",
    "qwen3vl_8b": "qwen",
    "qwen3vl_32b": "qwen32b",
    "internvl3_2b": "internvl2b",
    "internvl3_2b_local": "internvl2b",
    "internvl3_8b": "internvl",
    "internvl3_8b_local": "internvl",
    "internvl3_38b": "internvl38b",
    "llava_next_7b": "llava",
    "llava_next_7b_local": "llava",
    "llava_next_34b": "llava34b",
    "llava_med": "llavamed",
    "medgemma_4b": "medgemma4b",
    "lingshu_7b": "lingshu7b",
    "medvlm_r1": "medvlmr1",
}

MODEL_NUM_GPUS: Dict[str, int] = {
    "qwen3vl_2b": 1,
    "qwen3vl_4b_local": 1,
    "qwen3vl_8b": 1,
    "qwen3vl_32b": 1,
    "internvl3_2b": 1,
    "internvl3_2b_local": 1,
    "internvl3_8b": 1,
    "internvl3_8b_local": 1,
    "internvl3_38b": 2,
    "llava_next_7b": 1,
    "llava_next_7b_local": 1,
    "llava_next_34b": 1,
    "llava_med": 1,
    "medgemma_4b": 1,
    "lingshu_7b": 1,
    "medvlm_r1": 1,
}

MODEL_NUM_GPUS_4BIT: Dict[str, int] = {k: 1 for k in MODEL_NUM_GPUS}
