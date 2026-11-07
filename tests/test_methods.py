import numpy as np
import pytest
from scipy.special import softmax

from src.batch_calibration import batch_calibrate
from src.llm_shift_recalibration import bcts_transform, fit_bcts
from src.prior_shift_stress_test import prior_batch
from src.selective_prediction_under_shift import selective_metrics
from src.shift_recalibration import conformal_threshold, crc_threshold, ranking_auroc
from src.shift_triage import auroc_statistic, holm_survivors, survival_p_value


def test_conformal_coverage_is_valid_and_tight():
    rng = np.random.default_rng(0)
    alpha, n = 0.1, 49
    hits = []
    for _ in range(20000):
        s = rng.random(n + 1)
        hits.append(s[-1] <= conformal_threshold(s[:-1], alpha))
    coverage = np.mean(hits)
    assert 1 - alpha - 0.01 <= coverage <= 1 - alpha + 1 / (n + 1) + 0.01


def test_conformal_threshold_is_infinite_with_too_few_items():
    assert conformal_threshold(np.arange(5.0), 0.1) == np.inf


@pytest.mark.parametrize("n_pos,alpha", [(19, 0.05), (30, 0.1), (60, 0.05)])
def test_crc_false_negative_rate(n_pos, alpha):
    rng = np.random.default_rng(1)
    misses = []
    for _ in range(20000):
        s = rng.random(n_pos + 1)
        misses.append(s[-1] < crc_threshold(s[:-1], alpha))
    assert np.mean(misses) <= alpha + 0.01


def test_crc_flags_everything_below_minimum_positives():
    assert crc_threshold(np.random.default_rng(2).random(12), 0.05) == -np.inf


def test_bcts_without_bias_is_temperature_scaling():
    z = np.random.default_rng(3).normal(size=(10, 3))
    assert np.allclose(bcts_transform(z, np.array([np.log(2.0), 0.0, 0.0])), z / 2.0)


def test_bcts_recovers_temperature_and_bias():
    rng = np.random.default_rng(4)
    z = rng.normal(scale=3.0, size=(20000, 2))
    true = np.array([np.log(1.5), 0.8])
    y = np.array([rng.choice(2, p=p) for p in softmax(bcts_transform(z, true), axis=1)])
    theta, at_bound = fit_bcts(z, y)
    assert not at_bound
    assert np.allclose(theta, true, atol=0.05)


def test_map_bcts_with_tight_prior_returns_prior_mean():
    rng = np.random.default_rng(5)
    z, y = rng.normal(size=(50, 3)), rng.integers(0, 3, 50)
    mean = np.array([0.3, -0.2, 0.4])
    theta, _ = fit_bcts(z, y, prior_mean=mean, prior_var=(1e-8, 1e-8))
    assert np.allclose(theta, mean, atol=1e-3)


def test_batch_calibration_equalises_batch_marginals():
    p = softmax(np.random.default_rng(6).normal(size=(500, 3)) + [1.0, 0.0, -1.0], axis=1)
    q = batch_calibrate(p, p)
    assert np.allclose(q.sum(axis=1), 1.0)
    assert np.allclose((p / p.mean(axis=0)).mean(axis=0), 1.0)


def test_prior_batch_with_uniform_prior_equals_batch_calibration():
    rng = np.random.default_rng(7)
    p, batch = softmax(rng.normal(size=(40, 3)), axis=1), softmax(rng.normal(size=(200, 3)), axis=1)
    assert np.allclose(prior_batch(p, batch, np.full(3, 1 / 3)), batch_calibrate(p, batch))


def test_e_aurc_is_zero_for_a_perfect_ranking_and_nonnegative_otherwise():
    rng = np.random.default_rng(8)
    y = rng.integers(0, 2, 400)
    correct = rng.random(400) < 0.8
    pred = np.where(correct, y, 1 - y)
    conf = np.where(correct, 0.9, 0.6) + rng.random(400) * 0.01
    probs = np.zeros((400, 2))
    probs[np.arange(400), pred] = conf
    probs[np.arange(400), 1 - pred] = 1 - conf
    assert abs(selective_metrics(probs, y)["e_aurc"]) < 1e-9
    shuffled = probs[rng.permutation(400)]
    assert selective_metrics(shuffled, y)["e_aurc"] >= 0


@pytest.mark.parametrize("k", [2, 3])
def test_vectorised_auroc_matches_sklearn(k):
    rng = np.random.default_rng(9)
    y = rng.integers(0, k, 300)
    p = softmax(rng.normal(size=(300, k)) + np.eye(k)[y], axis=1)
    assert np.isclose(auroc_statistic(p, y, k)[0], ranking_auroc(p, y))


def test_survival_p_value_is_valid_under_the_null():
    rng = np.random.default_rng(10)
    p_values = []
    for _ in range(300):
        y = rng.integers(0, 2, 50)
        p = softmax(rng.normal(size=(50, 2)), axis=1)
        p_values.append(survival_p_value(p, y, rng))
    assert np.mean(np.array(p_values) <= 0.05) <= 0.05 + 0.03


def test_holm_survivors():
    assert holm_survivors({"a": 0.01, "b": 0.03}, 0.05) == {"a", "b"}
    assert holm_survivors({"a": 0.03, "b": 0.04}, 0.05) == set()
    assert holm_survivors({"a": 0.02, "b": 0.2}, 0.05) == {"a"}


@pytest.mark.parametrize("tau2,sigma2,n,delta", [(1, 0.5, 25, 0.5), (1, 0.5, 25, 1.2), (4, 0.2, 10, 0.9)])
def test_shrinkage_beats_mle_exactly_when_proposition_2_says_so(tau2, sigma2, n, delta):
    rng = np.random.default_rng(11)
    w = n * sigma2 / (n * sigma2 + tau2)
    mle = rng.normal(delta, np.sqrt(tau2 / n), 200000)
    better = np.mean((w * mle - delta) ** 2) < np.mean((mle - delta) ** 2)
    assert better == (delta ** 2 < 2 * sigma2 + tau2 / n)


def test_distance_aware_calibration_keeps_decisions_under_underflow():
    from scipy.special import softmax
    from sklearn.metrics import roc_auc_score

    from src.distance_aware_calibration import clip_like_metrics, dac, dac_metrics, dac_order_key
    from src.shift_recalibration import metrics

    rng = np.random.default_rng(0)
    for k in (2, 3):
        y = rng.integers(0, k, 2000)
        logits = rng.normal(0, 1, (2000, k))
        logits[np.arange(2000), y] += 1.5
        logits[:500] *= 12.0  # saturated probabilities, which differ from each other by about 1e-12
        p = softmax(logits, axis=1)
        d = rng.uniform(0.2, 1.4, 2000)
        assert dac_metrics(p, d, 0.0, y) == metrics(p, y)
        clipped = clip_like_metrics(p)
        for phi in (0.5, 2.0):
            exact, mixed = dac_metrics(p, d, phi, y), dac(clipped, d, phi)
            assert exact["macro_f1"] == metrics(p, y)["macro_f1"]
            assert abs(exact["nll"] - metrics(mixed, y)["nll"]) < 1e-9
            if k == 2:
                assert abs(exact["auroc"] - roc_auc_score(y, mixed[:, 1])) < 1e-12
            confidence = mixed.max(axis=1)
            ranks = dac_order_key(clipped.max(axis=1), d, phi, k)
            assert (np.argsort(confidence, kind="stable") == np.argsort(ranks, kind="stable")).all()
        assert dac_metrics(p, d, 1000.0, y)["macro_f1"] == metrics(p, y)["macro_f1"]
