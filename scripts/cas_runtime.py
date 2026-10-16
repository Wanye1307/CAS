"""Closed-answer CAS probes shared by the benchmark entry points."""

from __future__ import annotations

from typing import Any, Optional

from PIL import ImageFilter

from confidence.grounding import (
    CLOSED_GROUNDING_PROMPT,
    _binary_next_token_probs,
    _to_rgb_image,
    make_baseline_image,
)


def make_baseline(name: str, image: Any, *, seed: Optional[int]) -> Any:
    if name == "blur":
        return _to_rgb_image(image).filter(ImageFilter.GaussianBlur(radius=8.0))
    return make_baseline_image(image, baseline=name, seed=seed)


def run_binary(model: Any, processor: Any, model_config: Any,
               image: Optional[Any], question: str) -> dict[str, float]:
    prompt = CLOSED_GROUNDING_PROMPT.format(question=question)
    return _binary_next_token_probs(model, processor, model_config, image, prompt)
