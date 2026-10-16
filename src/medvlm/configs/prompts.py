"""Prompt templates for sampling-based and verbalized confidence estimation."""
from __future__ import annotations

from typing import Dict, List

VERBALIZED_VARIANTS: List[str] = [
    "vanilla", "vanilla_cot", "punish", "top_k", "two_stage", "linguistic",
]

VERBALIZED_PROMPTS: Dict[str, str] = {
    "vanilla": (
        "You are a medical AI assistant. Look at the provided medical image "
        "and answer the following question.\n\n"
        "Question: {question}\n\n"
        "Provide your answer and your confidence level (0-100%) in your answer.\n"
        "Format:\n"
        "Answer: [your answer]\n"
        "Confidence: [0-100]%"
    ),
    "vanilla_cot": (
        "You are a medical AI assistant. Look at the provided medical image "
        "and answer the following question.\n\n"
        "Question: {question}\n\n"
        "Think step by step. Then provide your answer and your confidence level "
        "(0-100%) in your answer.\n"
        "Format:\n"
        "Reasoning: [your reasoning]\n"
        "Answer: [your answer]\n"
        "Confidence: [0-100]%"
    ),
    "punish": (
        "You are a medical AI assistant. Look at the provided medical image "
        "and answer the following question. You will be punished if the answer "
        "is wrong but you answer it with high confidence.\n\n"
        "Question: {question}\n\n"
        "Provide your answer and your confidence level (0-100%) in your answer.\n"
        "Format:\n"
        "Answer: [your answer]\n"
        "Confidence: [0-100]%"
    ),
    "top_k": (
        "You are a medical AI assistant. Look at the provided medical image "
        "and answer the following question.\n\n"
        "Question: {question}\n\n"
        "Provide your top 3 best guesses for the answer, along with the "
        "probability (0-100%) that each guess is correct. The probabilities "
        "should sum to 100%.\n"
        "Format:\n"
        "Guess 1: [answer] (Probability: [X]%)\n"
        "Guess 2: [answer] (Probability: [Y]%)\n"
        "Guess 3: [answer] (Probability: [Z]%)"
    ),
    "two_stage_s1": (
        "You are a medical AI assistant. Look at the provided medical image "
        "and answer the following question.\n\n"
        "Question: {question}\n\n"
        "Provide your answer."
    ),
    "two_stage_s2": (
        "Question: {question}\n"
        "Proposed answer: {answer}\n\n"
        "How likely is the above answer to be correct? Provide a probability "
        "between 0% and 100%.\n"
        "Format:\n"
        "Confidence: [0-100]%"
    ),
    "linguistic": (
        "You are a medical AI assistant. Look at the provided medical image "
        "and answer the following question.\n\n"
        "Question: {question}\n\n"
        'Provide your answer and describe how confident you are using one of '
        'these terms: "almost certain", "highly likely", "very good chance", '
        '"probable", "likely", "better than even", "about even", "unlikely", '
        '"improbable", "very good chance not", "highly unlikely", '
        '"almost certainly not".\n'
        "Format:\n"
        "Answer: [your answer]\n"
        "Confidence: [one of the terms above]"
    ),
}

LINGUISTIC_MAP: Dict[str, float] = {
    "almost certain": 0.95,
    "highly likely": 0.90,
    "very good chance": 0.85,
    "probable": 0.75,
    "likely": 0.70,
    "better than even": 0.60,
    "about even": 0.50,
    "unlikely": 0.30,
    "improbable": 0.20,
    "very good chance not": 0.15,
    "highly unlikely": 0.10,
    "almost certainly not": 0.05,
}

BASE_SAMPLING_PROMPT: str = (
    "You are a medical AI assistant. Look at the provided medical image "
    "and answer the following question.\n\n"
    "Question: {question}\n\n"
    "Provide only the answer, without any explanation.\n"
    "Format:\n"
    "Answer: [your answer]"
)

COT_SAMPLING_PROMPT: str = (
    "You are a medical AI assistant. Look at the provided medical image "
    "and answer the following question.\n\n"
    "Question: {question}\n\n"
    "Think step by step. Then provide your answer.\n"
    "Format:\n"
    "Reasoning: [your reasoning]\n"
    "Answer: [your answer]"
)
