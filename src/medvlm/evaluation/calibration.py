"""
Calibration evaluation metrics for Medical VQA models.

Computes:
- ECE (Expected Calibration Error)
- MCE (Maximum Calibration Error)
- Overconfidence
- Accuracy
- Per-bin statistics
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np

from ..inference import VQAPrediction


@dataclass
class CalibrationResult:
    """Result of calibration evaluation for a single sample."""
    question: str
    ground_truth: str
    predicted: str
    confidence: float
    p_yes: float
    p_no: float
    is_correct: bool

    # Raw counts
    yes_count: int
    no_count: int
    unknown_count: int


def parse_yes_no(text: str) -> Optional[str]:
    """Parse yes/no from model response.

    Args:
        text: Model response text

    Returns:
        'yes', 'no', or None if unclear
    """
    text_lower = text.lower().strip()

    # Exact match
    if text_lower in ["yes", "yes.", "yes,", "yes!"]:
        return "yes"
    if text_lower in ["no", "no.", "no,", "no!"]:
        return "no"

    # Starts with
    if text_lower.startswith("yes"):
        return "yes"
    if text_lower.startswith("no"):
        return "no"

    # Contains (less reliable)
    if "yes" in text_lower and "no" not in text_lower:
        return "yes"
    if "no" in text_lower and "yes" not in text_lower:
        return "no"

    return None


def compute_empirical_probability(predictions: List[str]) -> CalibrationResult:
    """Compute empirical probability from multiple samples.

    Args:
        predictions: List of model predictions

    Returns:
        Tuple of (p_yes, p_no, yes_count, no_count, unknown_count)
    """
    yes_count = 0
    no_count = 0
    unknown_count = 0

    for pred in predictions:
        parsed = parse_yes_no(pred)
        if parsed == "yes":
            yes_count += 1
        elif parsed == "no":
            no_count += 1
        else:
            unknown_count += 1

    valid_count = yes_count + no_count

    if valid_count > 0:
        p_yes = yes_count / valid_count
        p_no = no_count / valid_count
    else:
        p_yes = 0.5
        p_no = 0.5

    return p_yes, p_no, yes_count, no_count, unknown_count


def evaluate_calibration_single(
    prediction: VQAPrediction,
) -> CalibrationResult:
    """Evaluate calibration for a single sample.

    Args:
        prediction: VQAPrediction with multiple samples

    Returns:
        CalibrationResult
    """
    p_yes, p_no, yes_count, no_count, unknown_count = compute_empirical_probability(
        prediction.predictions
    )

    # Determine prediction and confidence
    predicted = "yes" if p_yes >= 0.5 else "no"
    confidence = max(p_yes, p_no)

    # Check correctness
    gt_lower = prediction.ground_truth.lower().strip()
    is_correct = predicted == gt_lower

    return CalibrationResult(
        question=prediction.question,
        ground_truth=prediction.ground_truth,
        predicted=predicted,
        confidence=confidence,
        p_yes=p_yes,
        p_no=p_no,
        is_correct=is_correct,
        yes_count=yes_count,
        no_count=no_count,
        unknown_count=unknown_count,
    )


def compute_calibration_metrics(
    results: List[CalibrationResult],
    num_bins: int = 10,
) -> Dict[str, Any]:
    """Compute ECE, MCE, and overconfidence metrics.

    Args:
        results: List of CalibrationResult objects
        num_bins: Number of bins for ECE calculation

    Returns:
        Dictionary with metrics and bin data
    """
    confidences = np.array([r.confidence for r in results])
    accuracies = np.array([r.is_correct for r in results])

    # Bin boundaries
    bin_boundaries = np.linspace(0, 1, num_bins + 1)

    ece = 0.0
    mce = 0.0
    overconfidence = 0.0
    bin_data = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = in_bin.sum()

        if bin_size > 0:
            bin_acc = accuracies[in_bin].mean()
            bin_conf = confidences[in_bin].mean()

            # ECE contribution
            gap = abs(bin_acc - bin_conf)
            ece += (bin_size / len(results)) * gap

            # MCE
            mce = max(mce, gap)

            # Overconfidence (only when conf > acc)
            if bin_conf > bin_acc:
                overconfidence += (bin_size / len(results)) * (bin_conf - bin_acc)

            bin_data.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "bin_size": int(bin_size),
                "accuracy": float(bin_acc),
                "confidence": float(bin_conf),
                "gap": float(gap),
            })

    return {
        "ece": float(ece),
        "mce": float(mce),
        "overconfidence": float(overconfidence),
        "accuracy": float(accuracies.mean()),
        "mean_confidence": float(confidences.mean()),
        "num_samples": len(results),
        "bin_data": bin_data,
    }


class CalibrationEvaluator:
    """Evaluator for model calibration on closed questions."""

    def __init__(self, num_bins: int = 10):
        """Initialize evaluator.

        Args:
            num_bins: Number of bins for ECE calculation
        """
        self.num_bins = num_bins
        self.results: List[CalibrationResult] = []

    def add_prediction(self, prediction: VQAPrediction) -> CalibrationResult:
        """Add a prediction and compute its calibration result.

        Args:
            prediction: VQAPrediction object

        Returns:
            CalibrationResult
        """
        result = evaluate_calibration_single(prediction)
        self.results.append(result)
        return result

    def add_predictions(self, predictions: List[VQAPrediction]) -> None:
        """Add multiple predictions.

        Args:
            predictions: List of VQAPrediction objects
        """
        for pred in predictions:
            # Only evaluate closed questions
            if pred.answer_type == "closed":
                self.add_prediction(pred)

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute all calibration metrics.

        Returns:
            Dictionary with ECE, MCE, accuracy, etc.
        """
        if not self.results:
            raise ValueError("No results to evaluate. Add predictions first.")

        return compute_calibration_metrics(self.results, self.num_bins)

    def save_results(self, output_dir: str, prefix: str = "") -> None:
        """Save evaluation results to files.

        Args:
            output_dir: Directory to save results
            prefix: Optional prefix for filenames
        """
        os.makedirs(output_dir, exist_ok=True)

        # Compute metrics
        metrics = self.compute_metrics()

        # Save metrics summary
        metrics_path = os.path.join(output_dir, f"{prefix}metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save detailed results
        details_path = os.path.join(output_dir, f"{prefix}detailed_results.json")
        detailed = []
        for r in self.results:
            detailed.append({
                "question": r.question,
                "ground_truth": r.ground_truth,
                "predicted": r.predicted,
                "confidence": r.confidence,
                "p_yes": r.p_yes,
                "p_no": r.p_no,
                "is_correct": r.is_correct,
                "yes_count": r.yes_count,
                "no_count": r.no_count,
                "unknown_count": r.unknown_count,
            })

        with open(details_path, "w") as f:
            json.dump(detailed, f, indent=2)

        # Save human-readable summary
        summary_path = os.path.join(output_dir, f"{prefix}summary.txt")
        with open(summary_path, "w") as f:
            f.write("Calibration Evaluation Results\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Number of samples: {metrics['num_samples']}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Mean Confidence: {metrics['mean_confidence']:.4f}\n")
            f.write(f"ECE: {metrics['ece']:.4f}\n")
            f.write(f"MCE: {metrics['mce']:.4f}\n")
            f.write(f"Overconfidence: {metrics['overconfidence']:.4f}\n")
            f.write("\nBin Data:\n")
            f.write("-" * 40 + "\n")
            for bin_info in metrics['bin_data']:
                f.write(f"  [{bin_info['bin_lower']:.1f}-{bin_info['bin_upper']:.1f}]: "
                       f"n={bin_info['bin_size']}, "
                       f"acc={bin_info['accuracy']:.3f}, "
                       f"conf={bin_info['confidence']:.3f}, "
                       f"gap={bin_info['gap']:.3f}\n")

        print(f"Results saved to: {output_dir}")

    def print_summary(self) -> None:
        """Print evaluation summary to console."""
        metrics = self.compute_metrics()

        print("\n" + "=" * 50)
        print("CALIBRATION RESULTS")
        print("=" * 50)
        print(f"Samples: {metrics['num_samples']}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Mean Confidence: {metrics['mean_confidence']:.4f}")
        print(f"ECE: {metrics['ece']:.4f}")
        print(f"MCE: {metrics['mce']:.4f}")
        print(f"Overconfidence: {metrics['overconfidence']:.4f}")
        print("=" * 50)


# =============================================================================
# CalibrationPipeline: fit/transform for post-hoc calibration
# =============================================================================

class CalibrationPipeline:
    """Post-hoc calibration with fit/transform interface.

    Fits a calibration model on validation data and transforms
    confidence scores on test data.

    Standard methods (confidence only):
      - "temperature_scaling": Learns a single temperature T to scale logits.
      - "platt": Platt scaling on logit(c) = log(c/(1-c)) (textbook Platt).
      - "platt_confidence": Platt scaling on raw confidence c: sigma(a*c + b).
      - "isotonic": Isotonic regression (non-parametric).
      - "histogram_binning": Histogram binning calibration.

    HAC methods (confidence + hallucination scores):
      - "hac_platt": sigma(a*logit(c) + b*h + d), a>=0, b<=0 (regularized).
      - "hac_platt_confidence": Same but on raw c instead of logit(c).
      - "hac_gate": c * sigma(-a*h + b), a>=0 (regularized).

    Example::

        # Standard calibration
        cal = CalibrationPipeline(method="temperature_scaling")
        cal.fit(val_conf, val_correct)
        calibrated = cal.transform(test_conf)

        # HAC calibration (requires hallucination scores)
        cal = CalibrationPipeline(method="hac_platt")
        cal.fit(val_conf, val_correct, hallucination_scores=val_h)
        calibrated = cal.transform(test_conf, hallucination_scores=test_h)
    """

    STANDARD_METHODS = [
        "temperature_scaling", "platt", "platt_confidence",
        "isotonic", "histogram_binning",
    ]
    HAC_METHODS = ["hac_platt", "hac_platt_confidence", "hac_gate"]
    SUPPORTED_METHODS = STANDARD_METHODS + HAC_METHODS

    # L2 regularization on hallucination coefficient for HAC methods.
    HAC_LAMBDA = 0.01

    def __init__(self, method: str = "temperature_scaling", num_bins: int = 15):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unknown method: {method}. "
                           f"Available: {self.SUPPORTED_METHODS}")
        self.method = method
        self.num_bins = num_bins
        self._fitted = False
        self._params = {}

    @property
    def requires_hallucination_scores(self) -> bool:
        """Whether this method needs hallucination scores."""
        return self.method in self.HAC_METHODS

    def fit(self, confidences: np.ndarray, correctness: np.ndarray,
            hallucination_scores: Optional[np.ndarray] = None,
            ) -> "CalibrationPipeline":
        """Fit calibration model on validation data.

        Args:
            confidences: Array of confidence scores in [0, 1].
            correctness: Array of binary correctness labels (0 or 1).
            hallucination_scores: Array of hallucination/uncertainty scores
                (required for HAC methods; higher = more hallucinated).

        Returns:
            self (for method chaining).
        """
        confidences = np.asarray(confidences, dtype=np.float64)
        correctness = np.asarray(correctness, dtype=np.float64)

        if len(confidences) != len(correctness):
            raise ValueError("confidences and correctness must have the same length")

        if self.requires_hallucination_scores:
            if hallucination_scores is None:
                raise ValueError(
                    f"Method '{self.method}' requires hallucination_scores. "
                    f"Pass hallucination_scores=... to fit().")
            hallucination_scores = np.asarray(hallucination_scores, dtype=np.float64)
            if len(hallucination_scores) != len(confidences):
                raise ValueError("hallucination_scores must have the same length as confidences")

        if self.method == "temperature_scaling":
            self._fit_temperature_scaling(confidences, correctness)
        elif self.method == "platt":
            self._fit_platt(confidences, correctness, use_logit=True)
        elif self.method == "platt_confidence":
            self._fit_platt(confidences, correctness, use_logit=False)
        elif self.method == "isotonic":
            self._fit_isotonic(confidences, correctness)
        elif self.method == "histogram_binning":
            self._fit_histogram_binning(confidences, correctness)
        elif self.method == "hac_platt":
            self._fit_hac_platt(confidences, correctness, hallucination_scores,
                                lam=self.HAC_LAMBDA, use_logit=True)
        elif self.method == "hac_platt_confidence":
            self._fit_hac_platt(confidences, correctness, hallucination_scores,
                                lam=self.HAC_LAMBDA, use_logit=False)
        elif self.method == "hac_gate":
            self._fit_hac_gate(confidences, correctness, hallucination_scores,
                               lam=self.HAC_LAMBDA)

        self._fitted = True
        return self

    def transform(self, confidences: np.ndarray,
                  hallucination_scores: Optional[np.ndarray] = None,
                  ) -> np.ndarray:
        """Apply fitted calibration to confidence scores.

        Args:
            confidences: Array of confidence scores in [0, 1].
            hallucination_scores: Array of hallucination scores
                (required for HAC methods).

        Returns:
            Calibrated confidence scores.
        """
        if not self._fitted:
            raise RuntimeError("CalibrationPipeline not fitted. Call fit() first.")

        confidences = np.asarray(confidences, dtype=np.float64)

        if self.requires_hallucination_scores:
            if hallucination_scores is None:
                raise ValueError(
                    f"Method '{self.method}' requires hallucination_scores. "
                    f"Pass hallucination_scores=... to transform().")
            hallucination_scores = np.asarray(hallucination_scores, dtype=np.float64)

        if self.method == "temperature_scaling":
            return self._transform_temperature_scaling(confidences)
        elif self.method in ("platt", "platt_confidence"):
            return self._transform_platt(confidences)
        elif self.method == "isotonic":
            return self._transform_isotonic(confidences)
        elif self.method == "histogram_binning":
            return self._transform_histogram_binning(confidences)
        elif self.method in ("hac_platt", "hac_platt_confidence"):
            return self._transform_hac_platt(confidences, hallucination_scores)
        elif self.method == "hac_gate":
            return self._transform_hac_gate(confidences, hallucination_scores)

    def fit_transform(self, confidences: np.ndarray, correctness: np.ndarray,
                      hallucination_scores: Optional[np.ndarray] = None,
                      ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(confidences, correctness, hallucination_scores=hallucination_scores)
        return self.transform(confidences, hallucination_scores=hallucination_scores)

    # --- Temperature Scaling ---

    def _fit_temperature_scaling(self, confidences, correctness):
        """Learn optimal temperature T that minimizes NLL."""
        from scipy.optimize import minimize_scalar

        # Convert confidence to logits (inverse sigmoid)
        eps = 1e-7
        conf_clipped = np.clip(confidences, eps, 1 - eps)
        logits = np.log(conf_clipped / (1 - conf_clipped))

        def nll(T):
            scaled = 1.0 / (1.0 + np.exp(-logits / T))
            scaled = np.clip(scaled, eps, 1 - eps)
            return -np.mean(
                correctness * np.log(scaled) + (1 - correctness) * np.log(1 - scaled)
            )

        result = minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded")
        self._params["temperature"] = result.x

    def _transform_temperature_scaling(self, confidences):
        T = self._params["temperature"]
        eps = 1e-7
        conf_clipped = np.clip(confidences, eps, 1 - eps)
        logits = np.log(conf_clipped / (1 - conf_clipped))
        return 1.0 / (1.0 + np.exp(-logits / T))

    # --- Platt Scaling ---

    @staticmethod
    def _logit(c):
        c = np.clip(c, 1e-6, 1 - 1e-6)
        return np.log(c / (1 - c))

    def _fit_platt(self, confidences, correctness, use_logit=False):
        """Logistic regression on confidence (or logit(confidence))."""
        from sklearn.linear_model import LogisticRegression

        x = self._logit(confidences) if use_logit else confidences
        lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        lr.fit(x.reshape(-1, 1), correctness)
        self._params["platt_model"] = lr
        self._params["platt_use_logit"] = use_logit

    def _transform_platt(self, confidences):
        lr = self._params["platt_model"]
        x = self._logit(confidences) if self._params.get("platt_use_logit") else confidences
        return lr.predict_proba(x.reshape(-1, 1))[:, 1]

    # --- Isotonic Regression ---

    def _fit_isotonic(self, confidences, correctness):
        """Non-parametric isotonic regression."""
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso.fit(confidences, correctness)
        self._params["isotonic_model"] = iso

    def _transform_isotonic(self, confidences):
        iso = self._params["isotonic_model"]
        return iso.predict(confidences)

    # --- Histogram Binning ---

    def _fit_histogram_binning(self, confidences, correctness):
        """Simple histogram binning calibration."""
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_calibrated = np.zeros(self.num_bins)

        for i in range(self.num_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                bin_calibrated[i] = correctness[in_bin].mean()
            else:
                bin_calibrated[i] = (bin_boundaries[i] + bin_boundaries[i + 1]) / 2

        self._params["bin_boundaries"] = bin_boundaries
        self._params["bin_calibrated"] = bin_calibrated

    def _transform_histogram_binning(self, confidences):
        bin_boundaries = self._params["bin_boundaries"]
        bin_calibrated = self._params["bin_calibrated"]

        calibrated = np.zeros_like(confidences)
        for i in range(len(bin_calibrated)):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            calibrated[in_bin] = bin_calibrated[i]
        return calibrated

    # --- HAC-Platt: sigma(a*c + b*h + d), a >= 0, b <= 0 ---

    def _fit_hac_platt(self, c, y, h, lam=0.01, use_logit=False):
        from scipy.optimize import minimize as sp_minimize

        c_in = self._logit(c) if use_logit else c

        def nll(params):
            a, b, d = params
            logits = np.clip(a * c_in + b * h + d, -30, 30)
            p = np.clip(1.0 / (1.0 + np.exp(-logits)), 1e-12, 1 - 1e-12)
            loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            if lam > 0:
                loss += lam * b ** 2
            return loss

        res = sp_minimize(nll, x0=[1.0, -0.5, 0.0],
                          bounds=[(0, None), (None, 0), (None, None)],
                          method="L-BFGS-B")
        self._params["hac_platt_abd"] = res.x
        self._params["hac_platt_use_logit"] = use_logit

    def _transform_hac_platt(self, c, h):
        a, b, d = self._params["hac_platt_abd"]
        c_in = self._logit(c) if self._params.get("hac_platt_use_logit") else c
        logits = np.clip(a * c_in + b * h + d, -30, 30)
        return 1.0 / (1.0 + np.exp(-logits))

    # --- HAC-Gate: c * sigma(-a*h + b), a >= 0 ---

    def _fit_hac_gate(self, c, y, h, lam=0.01):
        from scipy.optimize import minimize as sp_minimize

        def nll(params):
            a, b = params
            gate = 1.0 / (1.0 + np.exp(-(-a * h + b)))
            c_adj = np.clip(c * gate, 1e-8, 1 - 1e-8)
            loss = -np.mean(y * np.log(c_adj) + (1 - y) * np.log(1 - c_adj))
            if lam > 0:
                loss += lam * a ** 2
            return loss

        res = sp_minimize(nll, x0=[1.0, 0.0],
                          bounds=[(0, None), (None, None)],
                          method="L-BFGS-B")
        self._params["hac_gate_ab"] = res.x

    def _transform_hac_gate(self, c, h):
        a, b = self._params["hac_gate_ab"]
        gate = 1.0 / (1.0 + np.exp(-(-a * h + b)))
        return np.clip(c * gate, 0, 1)
