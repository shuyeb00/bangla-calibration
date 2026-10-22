import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import softmax


def compute_nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    n = len(labels)
    true_class_probs = np.clip(probs[np.arange(n), labels], eps, 1.0)
    return float(-np.mean(np.log(true_class_probs)))


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            ece += np.abs(np.mean(accuracies[in_bin]) - np.mean(confidences[in_bin])) * prop_in_bin
    return float(ece)


def compute_mce(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if np.sum(in_bin) > 0:
            mce = max(mce, float(np.abs(np.mean(accuracies[in_bin]) - np.mean(confidences[in_bin]))))
    return float(mce)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    n = len(labels)
    one_hot = np.zeros((n, n_classes), dtype=float)
    one_hot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 50) -> float:
    from scipy.optimize import minimize_scalar
    from scipy.special import log_softmax
    logits_f = np.asarray(logits, dtype=np.float64)
    labels_i = np.asarray(labels, dtype=np.int64)

    def nll_obj(t: float) -> float:
        log_probs = log_softmax(logits_f / max(t, 1e-3), axis=1)
        return float(-np.mean(log_probs[np.arange(len(labels_i)), labels_i]))

    res = minimize_scalar(nll_obj, bounds=(0.05, 10.0), method='bounded', options={'maxiter': max_iter})
    return float(np.clip(res.x, 0.05, 10.0))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(logits / temperature, axis=1)


def risk_coverage_curve(probs: np.ndarray, labels: np.ndarray):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels).astype(float)
    order = np.argsort(-confidences)
    sorted_correct = correct[order]
    n = len(labels)
    coverages = np.arange(1, n + 1) / n
    risks = np.cumsum(1.0 - sorted_correct) / np.arange(1, n + 1)
    return coverages, risks


def compute_aurc(coverages: np.ndarray, risks: np.ndarray) -> float:
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapz_fn(risks, coverages))


def risk_at_coverage(coverages: np.ndarray, risks: np.ndarray, target_coverage: float) -> float:
    idx = np.searchsorted(coverages, target_coverage)
    return float(risks[min(idx, len(risks) - 1)])


def conformal_calibrate(cal_probs: np.ndarray, cal_labels: np.ndarray, alpha: float = 0.1) -> float:
    n = len(cal_labels)
    true_class_probs = cal_probs[np.arange(n), cal_labels]
    scores = 1.0 - true_class_probs
    sorted_scores = np.sort(scores)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(sorted_scores[k - 1])


def build_prediction_sets(test_probs: np.ndarray, q_hat: float) -> list:
    n, k = test_probs.shape
    prediction_sets = []
    for i in range(n):
        included = [c for c in range(k) if (1.0 - test_probs[i, c]) <= q_hat]
        if not included:
            included = [int(np.argmax(test_probs[i]))]
        prediction_sets.append(included)
    return prediction_sets


def evaluate_conformal(test_probs: np.ndarray, test_labels: np.ndarray, q_hat: float) -> dict:
    pred_sets = build_prediction_sets(test_probs, q_hat)
    covered = [test_labels[i] in pred_sets[i] for i in range(len(test_labels))]
    return {
        "empirical_coverage": float(np.mean(covered)),
        "avg_set_size": float(np.mean([len(s) for s in pred_sets])),
    }