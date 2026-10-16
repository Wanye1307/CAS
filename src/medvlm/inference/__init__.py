"""Batch inference and multi-sample generation for VQA models."""
from .predictor import VQAPrediction, VQAInference, run_inference
__all__ = ["VQAPrediction", "VQAInference", "run_inference"]
