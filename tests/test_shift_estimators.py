import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from src.batch_calibration import batch_calibrate
from src.bcts_shift_coordinates import batch_bias, projection
from src.label_shift_estimators import bbse_prior
from src.llm_shift_recalibration import em_label_shift
from src.operating_point_analysis import fit_bias
from src.prior_drift_stress_test import balanced_accuracy
from src.translation_audit import log_odds_z, select_terms, words
from src.triage_selection_sensitivity import class_aurocs, delong_logit_lcb, hoeffding_lcb


def binormal(rng, n, auroc, prior=0.5):
    y = (rng.random(n) < prior).astype(int)
    z = rng.normal(np.sqrt(2) * norm.ppf(auroc) * y, 1.0)
    return y, np.c_[np.zeros(n), z]


@pytest.mark.parametrize("k", [2, 3])
def test_class_aurocs_match_sklearn(k):
    rng = np.random.default_rng(0)
    y = rng.integers(0, k, 200)
    p = softmax(rng.normal(0, 1, (200, k)) + 1.2 * np.eye(k)[y], axis=1)
    ours = np.mean([a for a, _, _ in class_aurocs(p, y, k)])
    ref = roc_auc_score(y, p[:, 1]) if k == 2 else roc_auc_score(y, p, multi_class="ovr")
    assert ours == pytest.approx(ref, abs=1e-12)


@pytest.mark.parametrize("auroc", [0.7, 0.9])
def test_lower_bounds_cover_the_population_auroc(auroc):
    rng = np.random.default_rng(1)
    hits_h, hits_d, reps = 0, 0, 1500
    for _ in range(reps):
        y, s = binormal(rng, 50, auroc)
        hits_h += hoeffding_lcb(s, y, 0.025) <= auroc
        hits_d += delong_logit_lcb(s, y, 0.025) <= auroc
    assert hits_h / reps >= 0.975
    assert hits_d / reps >= 0.965


def test_lower_bounds_need_both_classes():
    s = np.c_[np.zeros(10), np.arange(10.0)]
    assert hoeffding_lcb(s, np.ones(10, int), 0.025) == -np.inf
    assert delong_logit_lcb(s, np.ones(10, int), 0.025) == -np.inf


def test_bbse_and_em_recover_a_pure_label_shift():
    rng = np.random.default_rng(2)
    def draw(n, prior):
        y = rng.choice(2, n, p=prior)
        s = rng.normal(1.5 * y, 1.0)
        logit = 1.5 * s - 1.125
        return y, np.c_[1 - 1 / (1 + np.exp(-logit)), 1 / (1 + np.exp(-logit))]
    y_s, p_s = draw(20000, [0.5, 0.5])
    _, p_t = draw(20000, [0.85, 0.15])
    assert bbse_prior(p_s.argmax(1), y_s, p_t.argmax(1), 2)[1] == pytest.approx(0.15, abs=0.04)
    assert em_label_shift(p_t, np.array([0.5, 0.5]))[1] == pytest.approx(0.15, abs=0.02)


def test_batch_bias_reproduces_batch_calibration():
    rng = np.random.default_rng(3)
    p = softmax(rng.normal(0, 1, (500, 3)) + np.r_[0.0, 1.0, -0.5], axis=1)
    assert np.allclose(softmax(np.log(p) + np.r_[0.0, batch_bias(p)], axis=1), batch_calibrate(p, p))


def test_projection_of_the_oracle_bias_onto_itself_is_one():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 2000)
    s = np.c_[np.zeros(2000), rng.normal(2.0 * y - 1.0, 1.0)] - np.r_[0.0, 1.5]
    b = fit_bias(s, y)
    assert projection(b, b) == pytest.approx(1.0)
    assert np.isnan(projection(b, np.array([0.05])))


def test_balanced_accuracy_ignores_absent_classes():
    y = np.array([0, 0, 1, 1])
    assert balanced_accuracy(np.array([0, 1, 1, 1]), y, 3) == pytest.approx(0.75)


def test_bangla_words_keep_their_vowel_signs():
    assert words("আমি ভালো আছি") == ["আমি", "ভালো", "আছি"]


def test_log_odds_selects_label_words_and_matched_controls():
    rng = np.random.default_rng(5)
    texts, labels = [], []
    for _ in range(3000):
        y = int(rng.random() < 0.4)
        toks = list(rng.choice([f"w{i}" for i in range(300)], 8))
        if y:
            toks += list(rng.choice([f"bad{i}" for i in range(20)], 2))
        texts.append(" ".join(toks))
        labels.append(y)
    z = log_odds_z(pd.Series(texts), np.array(labels))
    offensive, control = select_terms(z)
    assert {f"bad{i}" for i in range(20)} <= offensive
    assert not any(w.startswith("bad") for w in control)
    assert len(offensive) == len(control)
