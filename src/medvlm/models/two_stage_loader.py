"""
Two-stage adapter loading for contrastive GRPO → SFT adaptation models.

When a checkpoint directory contains a `loading_info.json` file, it indicates
that the adapter was trained on top of a merged contrastive GRPO adapter.
Loading it correctly requires:
  1. Load base model
  2. Load stage1 (contrastive GRPO) adapter → merge_and_unload()
  3. Load stage2 (SFT adaptation) adapter on the merged model

This module provides utilities to detect and handle this two-stage loading
transparently, so evaluation scripts don't need special-casing everywhere.
"""

import os
import json
from typing import Optional, Tuple, Any

import torch
from peft import PeftModel

from medvlm.configs import ModelConfig
from medvlm.models import load_model


LOADING_INFO_FILENAME = "loading_info.json"


def is_two_stage_adapter(adapter_path: Optional[str]) -> bool:
    """Check if an adapter path requires two-stage loading.

    Returns True if the adapter directory contains a loading_info.json file.
    """
    if adapter_path is None:
        return False
    info_path = os.path.join(adapter_path, LOADING_INFO_FILENAME)
    return os.path.isfile(info_path)


def get_loading_info(adapter_path: str) -> Optional[dict]:
    """Read the loading_info.json from an adapter directory.

    Returns the parsed JSON dict, or None if the file doesn't exist.
    """
    info_path = os.path.join(adapter_path, LOADING_INFO_FILENAME)
    if not os.path.isfile(info_path):
        return None
    with open(info_path) as f:
        return json.load(f)


def load_two_stage_model(
    model_config: ModelConfig,
    adapter_path: str,
) -> Tuple[Any, Any]:
    """Load a model with two-stage adapter loading.

    Reads loading_info.json from adapter_path to discover the stage1
    (contrastive GRPO) adapter, loads and merges it, then loads the
    stage2 (SFT adaptation) adapter on top.

    Args:
        model_config: Base model configuration
        adapter_path: Path to the stage2 adapter (must contain loading_info.json)

    Returns:
        Tuple of (model, processor)

    Raises:
        FileNotFoundError: If loading_info.json or stage1 adapter not found
        ValueError: If loading_info.json is malformed
    """
    info = get_loading_info(adapter_path)
    if info is None:
        raise FileNotFoundError(
            f"No {LOADING_INFO_FILENAME} found in {adapter_path}. "
            f"Use load_model() for standard single-adapter loading."
        )

    stage1_path = info.get("stage1_adapter_path")
    if stage1_path is None:
        raise ValueError(
            f"{LOADING_INFO_FILENAME} in {adapter_path} is missing 'stage1_adapter_path'"
        )

    # Verify stage1 adapter exists
    if not os.path.isdir(stage1_path):
        raise FileNotFoundError(
            f"Stage-1 adapter not found: {stage1_path}\n"
            f"Referenced by: {os.path.join(adapter_path, LOADING_INFO_FILENAME)}"
        )

    print(f"[Two-Stage Loading] Detected two-stage adapter")
    print(f"  Stage 1 (contrastive GRPO): {stage1_path}")
    print(f"  Stage 2 (SFT adaptation):   {adapter_path}")

    # Step 1: Load base model
    print(f"\n  [1/3] Loading base model: {model_config.model_id}")
    model, processor = load_model(
        model_config,
        adapter_path=None,  # No adapter yet
    )

    # Step 2: Load stage1 adapter and merge
    print(f"  [2/3] Loading & merging stage-1 adapter...")
    model = PeftModel.from_pretrained(model, stage1_path)
    model = model.merge_and_unload()
    print(f"         Stage-1 merged into base weights")

    if torch.cuda.is_available():
        print(f"         VRAM after merge: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Step 3: Load stage2 adapter
    print(f"  [3/3] Loading stage-2 adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    print(f"         Two-stage loading complete")

    return model, processor


def smart_load_model(
    model_config: ModelConfig,
    adapter_path: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Smart model loader that auto-detects two-stage adapters.

    Drop-in replacement for load_model() that transparently handles
    both standard single-adapter and two-stage (contrastive → SFT) loading.

    Args:
        model_config: Base model configuration
        adapter_path: Optional path to adapter (single or two-stage)

    Returns:
        Tuple of (model, processor)
    """
    if adapter_path is not None and is_two_stage_adapter(adapter_path):
        return load_two_stage_model(
            model_config, adapter_path
        )
    else:
        return load_model(
            model_config,
            adapter_path=adapter_path,
        )
