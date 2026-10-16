import numpy as np

if not hasattr(np, "infty"):
    np.infty = np.inf

from scipy.special import softmax
from sklearn.metrics import f1_score, accuracy_score
from netcal.scaling import TemperatureScaling
from netcal.metrics import ECE

def fit_temperature_scaling(cal_logits: np.ndarray, cal_labels: np.ndarray) -> TemperatureScaling:
    cal_probs = softmax(cal_logits, axis=1)
    ts = TemperatureScaling()
    ts.fit(cal_probs, cal_labels)
    return ts

def compute_conformal_threshold(cal_probs_scaled: np.ndarray, cal_labels: np.ndarray, alpha: float = 0.10) -> float:
    n_cal = len(cal_labels)
    cal_scores = 1.0 - cal_probs_scaled[np.arange(n_cal), cal_labels]
    q_level = min(np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal, 1.0)
    return float(np.quantile(cal_scores, q_level, method="higher"))

def evaluate_predictions(logits: np.ndarray, labels: np.ndarray, ts: TemperatureScaling, q_hat: float) -> dict:
    probs = softmax(logits, axis=1)
    probs_scaled = ts.transform(probs)
    
    preds = np.argmax(probs_scaled, axis=1)
    macro_f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    
    ece_metric = ECE(bins=15)
    ece_uncal = ece_metric.measure(probs, labels)
    ece_cal = ece_metric.measure(probs_scaled, labels)
    
    pred_sets = probs_scaled >= (1.0 - q_hat)
    coverage = np.mean(pred_sets[np.arange(len(labels)), labels])
    avg_set_size = np.mean(np.sum(pred_sets, axis=1))
    
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "ece_uncalibrated": round(float(ece_uncal), 4),
        "ece_calibrated": round(float(ece_cal), 4),
        "conformal_coverage_90": round(float(coverage), 4),
        "avg_set_size": round(float(avg_set_size), 4)
    }