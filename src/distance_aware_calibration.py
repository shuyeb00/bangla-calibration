"""Distance-aware calibration (DAC; Gasparin and Detommaso, Findings of EMNLP 2024) as a label-free baseline under shift.

DAC mixes a classifier's probabilities with the uniform distribution according to the distance of an input from the
training data in representation space:
    p_dac(y | x) = p(y | x) exp(-phi d(x)) + (1 - exp(-phi d(x))) / K,
with d(x) the k-nearest-neighbour distance (deep kNN; Sun et al., 2022) of the [CLS] embedding. phi is chosen on a
grid by trading off two objectives with the Kneedle knee (Satopaa et al., 2011): 1 - PRAUC of separating in-domain
(positive) from out-of-distribution (negative) calibration inputs by top-label confidence, and the in-domain ECE. Only
the membership of the calibration inputs is used, no target label, so DAC is label-free on the target like
importance-weighted TS and batch calibration. Because the mixing weight is a scalar per input, DAC never changes the
predicted class of an input; it can change the ranking of P(class) across inputs.

Implementation for this study (base grid, converged cross-entropy runs, both fine-tuning regimes):
  embeddings   last-layer [CLS] hidden states saved by src/train.py, L2-normalised (as in deep kNN)
  reference    the paper uses the training split; training-split hidden states are not stored, so the reference bank is
               one half of the source calibration split (stratified by label, REFERENCE_SEED) and the other half is the
               in-domain calibration set of the two objectives, so in-domain distances are out-of-sample
  distance     Euclidean distance to the K_NEIGHBOURS-th nearest reference embedding
  base model   source-TS probabilities (temperature fitted on the whole source calibration split)
  OOD set      the unlabelled target calibration inputs of each condition (the condition's own target)
  grid         phi in {0} and 61 log-spaced values in [1e-2, 1e3] (phi_grid); Kneedle on the Pareto front of (ECE, 1 - PRAUC), both normalised to [0, 1],
               knee = argmax(1 - ECE_n - (1 - PRAUC)_n) (convex decreasing curve, sensitivity 1, no smoothing)
Compared on the same runs and full target test splits: source TS and label-free batch calibration (unlabelled target
calibration batch, src/batch_calibration.py). Regimes are those of the regime map (publication_tables.regime_assignment).

Numerics: for large phi d the weight underflows and p_dac rounds to exactly 1/K for every class, which would turn
argmax into a tie broken towards class 0 and tie the rankings. In exact arithmetic p_dac - 1/K = w (p - 1/K), so the
decisions are those of p, and every ranking (confidence for AURC and PRAUC, class scores for AUROC) uses the exact
order of log w + log|p - 1/K| with the sign of p - 1/K (dac_order_key, dense ranks). NLL, ECE and Brier use p_dac.
DAC is applied to the probabilities after the clipping and renormalisation that shift_recalibration.metrics applies
to every method (clip_like_metrics), so at phi = 0 DAC equals source TS exactly and all methods share one convention.

Requires the hidden states of python -m src.fetch_run_outputs --hidden-target-splits.

Tables (results/tables/grid_max_length_512/):
  distance_aware_calibration_per_run.csv   per run, condition and method: metrics, chosen phi, distance statistics
  distance_aware_calibration_summary.csv   per task, condition (with regime) and method: means over runs, and the share
                                            of runs in which DAC beats source TS on NLL and macro-F1

    python -m src.distance_aware_calibration
    python -m src.distance_aware_calibration --natural-codemixed   (held-out natural code-mixed targets only)
    python -m src.distance_aware_calibration --phi-max-exponent 5   (sensitivity: phi grid up to 1e5; tables carry the
                                                                    suffix _phi_max_1e5)
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import average_precision_score

from sklearn.metrics import f1_score, roc_auc_score

from src.batch_calibration import batch_calibrate
from src.calibration import compute_aurc, compute_brier_score, compute_ece, compute_nll
from src.config import TASKS, scoped_table_name, use_natural_codemixed_scope
from src.llm_evaluation import GRID_DIR, TABLES, encoder_runs, split_labels
from src.shift_aware_routing import target_conditions
from src.shift_recalibration import EPS, fit_log_temperature, metrics

K_NEIGHBOURS = 50
PHI_MAX_EXPONENT = 3
PHI_POINTS_PER_DECADE = 12
REFERENCE_SEED = 20261107
OUTCOMES = ("accuracy", "macro_f1", "nll", "ece_15", "brier", "aurc", "auroc")
ECE_BINS = 15


def normalise(h: np.ndarray) -> np.ndarray:
    return h / np.clip(np.linalg.norm(h, axis=1, keepdims=True), 1e-12, None)


def knn_distance(queries: np.ndarray, bank: np.ndarray, k: int = K_NEIGHBOURS, chunk: int = 2048) -> np.ndarray:
    """Euclidean distance to the k-th nearest bank vector (unit vectors: d^2 = 2 - 2 cos)."""
    k = min(k, len(bank))
    out = np.empty(len(queries))
    for start in range(0, len(queries), chunk):
        cos = queries[start:start + chunk] @ bank.T
        kth = -np.partition(-cos, k - 1, axis=1)[:, k - 1]
        out[start:start + chunk] = np.sqrt(np.clip(2.0 - 2.0 * kth, 0.0, None))
    return out


def dac(probs: np.ndarray, distance: np.ndarray, phi: float) -> np.ndarray:
    w = np.exp(-phi * distance)[:, None]
    return probs * w + (1.0 - w) / probs.shape[1]


def clip_like_metrics(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, EPS, 1.0)
    return clipped / clipped.sum(axis=1, keepdims=True)


def _dense_rank(primary: np.ndarray, sign: np.ndarray, value: np.ndarray) -> np.ndarray:
    keys = np.empty(len(primary), dtype=[("primary", "f8"), ("sign", "i1"), ("value", "f8")])
    keys["primary"], keys["sign"], keys["value"] = primary, sign, value
    return np.unique(keys, return_inverse=True)[1].reshape(-1).astype(np.float64)


def dac_order_key(values: np.ndarray, distance: np.ndarray, phi: float, k: int) -> np.ndarray:
    """Dense ranks of 1/K + w (values - 1/K), w = exp(-phi d): ordered by the value computed in floating point and,
    where that value ties (w so small that the sum rounds to 1/K), by the sign of values - 1/K and then by
    +-(log w + log|values - 1/K|), which is the exact order of the sum. Only the order is meaningful."""
    w = np.exp(-phi * distance)
    log_w = -phi * distance
    if values.ndim == 2:
        w, log_w = w[:, None], log_w[:, None]
    primary = values * w + (1.0 - w) / k
    centred = values - 1.0 / k
    with np.errstate(divide="ignore"):
        magnitude = log_w + np.log(np.abs(centred))
    sign = np.sign(centred).astype(np.int8)
    value = np.where(sign > 0, magnitude, np.where(sign < 0, -magnitude, 0.0))
    if values.ndim == 1:
        return _dense_rank(primary, sign, value)
    return np.stack([_dense_rank(primary[:, c], sign[:, c], value[:, c]) for c in range(values.shape[1])], axis=1)


def dac_metrics(probs: np.ndarray, distance: np.ndarray, phi: float, labels: np.ndarray) -> dict:
    """metrics() of the DAC probabilities, with the decisions of the base probabilities (which DAC never changes) and
    rankings that stay exact where the mixed probabilities round to 1/K."""
    probs = clip_like_metrics(probs)
    k = probs.shape[1]
    mixed = dac(probs, distance, phi)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)
    order = np.argsort(-dac_order_key(probs.max(axis=1), distance, phi, k), kind="stable")
    n = len(labels)
    risks = np.cumsum(1.0 - correct[order]) / np.arange(1, n + 1)
    keys = dac_order_key(probs, distance, phi, k)
    if len(np.unique(labels)) < k:
        auroc = float("nan")
    elif k == 2:
        auroc = float(roc_auc_score(labels, keys[:, 1]))
    else:
        auroc = float(np.mean([roc_auc_score(labels == c, keys[:, c]) for c in range(k)]))
    return {"accuracy": float(correct.mean()),
            "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "nll": compute_nll(mixed, labels), "ece_15": compute_ece(mixed, labels, n_bins=15),
            "brier": compute_brier_score(mixed, labels, n_classes=k),
            "aurc": compute_aurc(np.arange(1, n + 1) / n, risks), "auroc": auroc}


def reference_split(labels: np.ndarray) -> tuple:
    """Stratified halves of the source calibration split: (reference bank, in-domain calibration set)."""
    rng = np.random.default_rng(REFERENCE_SEED)
    bank, ind = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        bank.append(idx[: len(idx) // 2])
        ind.append(idx[len(idx) // 2:])
    return np.sort(np.concatenate(bank)), np.sort(np.concatenate(ind))


def kneedle(ece: np.ndarray, one_minus_prauc: np.ndarray) -> int:
    """Index of the knee on the Pareto front of the two objectives (both to be minimised)."""
    order = np.lexsort((one_minus_prauc, ece))
    front, best = [], np.inf
    for i in order:
        if one_minus_prauc[i] < best - 1e-12:
            front.append(i)
            best = one_minus_prauc[i]
    front = np.array(front)
    if len(front) == 1:
        return int(front[0])
    x, y = ece[front], one_minus_prauc[front]
    xn = (x - x.min()) / max(x.max() - x.min(), 1e-12)
    yn = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    return int(front[np.argmax(1.0 - xn - yn)])


def phi_grid(max_exponent: int = PHI_MAX_EXPONENT) -> np.ndarray:
    return np.r_[0.0, np.logspace(-2, max_exponent, PHI_POINTS_PER_DECADE * (max_exponent + 2) + 1)]


def select_phi(p_ind: np.ndarray, y_ind: np.ndarray, d_ind: np.ndarray, p_ood: np.ndarray, d_ood: np.ndarray,
               max_exponent: int = PHI_MAX_EXPONENT) -> tuple:
    grid = phi_grid(max_exponent)
    p_ind, p_ood = clip_like_metrics(p_ind), clip_like_metrics(p_ood)
    membership = np.r_[np.ones(len(p_ind)), np.zeros(len(p_ood))]
    ece, one_minus = np.empty(len(grid)), np.empty(len(grid))
    for i, phi in enumerate(grid):
        k = p_ind.shape[1]
        confidence = np.r_[dac_order_key(p_ind.max(1), d_ind, phi, k), dac_order_key(p_ood.max(1), d_ood, phi, k)]
        one_minus[i] = 1.0 - average_precision_score(membership, confidence)
        ece[i] = compute_ece(dac(p_ind, d_ind, phi), y_ind, n_bins=ECE_BINS)
    j = kneedle(ece, one_minus)
    return float(grid[j]), float(ece[j]), float(1.0 - one_minus[j])


def process_run(run: dict, gold: dict, natural_codemixed: bool, max_exponent: int = PHI_MAX_EXPONENT) -> list:
    d, task = run["dir"], run["task"]
    if not (d / "cal_hidden.npy").exists():
        return []
    y_src = gold[task]["cal"]
    logits_src = np.load(d / "cal_logits.npy").astype(np.float64)
    h_src = normalise(np.load(d / "cal_hidden.npy").astype(np.float64))
    if len(h_src) != len(y_src) or len(logits_src) != len(y_src):
        raise ValueError(f"{d.name}: calibration outputs do not match the labels")
    t = float(np.exp(fit_log_temperature(logits_src, y_src)))
    bank_idx, ind_idx = reference_split(y_src)
    bank = h_src[bank_idx]
    p_ind = softmax(logits_src[ind_idx] / t, axis=1)
    d_ind = knn_distance(h_src[ind_idx], bank)
    key = {"run_id": d.name, "task": task, "model": run["model"],
           "ft_method": d.name.split(f"{run['model']}_", 1)[1].split("_seed")[0],
           "seed": int(d.name.rsplit("seed", 1)[1]), "temperature": t}
    rows = []
    for cond in target_conditions(task, natural_codemixed):
        cal = cond[: -len("_test")] + "_cal"
        needed = [d / f"{s}_{kind}.npy" for s in (cond, cal) for kind in ("logits", "hidden")]
        if not all(p.exists() for p in needed):
            continue
        y_t = gold[task][cond]
        p_t = softmax(np.load(d / f"{cond}_logits.npy").astype(np.float64) / t, axis=1)
        p_c = softmax(np.load(d / f"{cal}_logits.npy").astype(np.float64) / t, axis=1)
        d_t = knn_distance(normalise(np.load(d / f"{cond}_hidden.npy").astype(np.float64)), bank)
        d_c = knn_distance(normalise(np.load(d / f"{cal}_hidden.npy").astype(np.float64)), bank)
        if len(p_t) != len(y_t) or len(d_t) != len(y_t):
            raise ValueError(f"{d.name}: {cond} outputs do not match the labels")
        phi, ece_ind, prauc = select_phi(p_ind, y_src[ind_idx], d_ind, p_c, d_c, max_exponent)
        base = {**key, "condition": cond, "n_items": len(y_t), "phi": phi,
                "phi_at_upper_bound": bool(phi >= 10.0 ** max_exponent), "ind_ece_at_phi": ece_ind,
                "membership_prauc_at_phi": prauc, "distance_ind_median": float(np.median(d_ind)),
                "distance_target_median": float(np.median(d_t)),
                "mixing_weight_target_mean": float(np.exp(-phi * d_t).mean())}
        rows.append({**base, "method": "source_ts", **metrics(p_t, y_t)})
        rows.append({**base, "method": "dac", **dac_metrics(p_t, d_t, phi, y_t)})
        rows.append({**base, "method": "batch_calibration", **metrics(batch_calibrate(p_t, p_c), y_t)})
    return rows


def summarise(per_run: pd.DataFrame) -> pd.DataFrame:
    from src.publication_tables import regime_assignment

    regimes = regime_assignment()
    wide = per_run.pivot_table(index=["run_id", "task", "condition"], columns="method", values=["nll", "macro_f1",
                                                                                                "ece_15", "auroc"])
    wins = pd.DataFrame({
        "dac_nll_win": (wide["nll"]["dac"] < wide["nll"]["source_ts"] - 1e-12),
        "dac_macro_f1_change": wide["macro_f1"]["dac"] - wide["macro_f1"]["source_ts"],
        "dac_auroc_change": wide["auroc"]["dac"] - wide["auroc"]["source_ts"],
        "dac_vs_batch_nll": wide["nll"]["dac"] - wide["nll"]["batch_calibration"]}).reset_index()
    rows = []
    for (task, cond, method), g in per_run.groupby(["task", "condition", "method"]):
        w = wins[(wins.task == task) & (wins.condition == cond)]
        row = {"task": task, "condition": cond, "regime": regimes.get((task, cond), "unassigned"),
               "method": method, "n_runs": g.run_id.nunique(), **{c: g[c].mean() for c in OUTCOMES},
               "phi_median": g.phi.median(), "phi_at_upper_bound_share": g.phi_at_upper_bound.mean(),
               "phi_zero_share": (g.phi == 0).mean(),
               "mixing_weight_target_mean": g.mixing_weight_target_mean.mean()}
        if method == "dac":
            row.update({"dac_nll_win_share": w.dac_nll_win.mean(),
                        "dac_macro_f1_change_max_abs": w.dac_macro_f1_change.abs().max(),
                        "dac_auroc_change_mean": w.dac_auroc_change.mean(),
                        "dac_minus_batch_nll_mean": w.dac_vs_batch_nll.mean()})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distance-aware calibration as a label-free baseline under shift.")
    parser.add_argument("--natural-codemixed", action="store_true",
                        help="evaluate only the held-out natural code-mixed targets")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--phi-max-exponent", type=int, default=PHI_MAX_EXPONENT,
                        help="largest phi of the grid is 10**this; non-default values write suffixed tables")
    args = parser.parse_args()
    use_natural_codemixed_scope(args.natural_codemixed)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    start = time.time()
    gold = {task: split_labels(task) for task in TASKS}
    runs = encoder_runs()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_run, run, gold, args.natural_codemixed, args.phi_max_exponent): run["dir"].name
                   for run in runs}
        for i, future in enumerate(as_completed(futures)):
            rows += future.result()
            print(f"[{i + 1}/{len(runs)}] {futures[future]} ({time.time() - start:.0f} s)", flush=True)
    if not rows:
        raise SystemExit("No run has hidden states; run python -m src.fetch_run_outputs --hidden-target-splits first")
    per_run = pd.DataFrame(rows).sort_values(["task", "condition", "run_id", "method"]).reset_index(drop=True)
    summary = summarise(per_run).sort_values(["task", "condition", "method"]).reset_index(drop=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.phi_max_exponent == PHI_MAX_EXPONENT else f"_phi_max_1e{args.phi_max_exponent}"
    for name, table in (("distance_aware_calibration_per_run.csv", per_run),
                        ("distance_aware_calibration_summary.csv", summary)):
        out = TABLES / scoped_table_name(name.replace(".csv", f"{suffix}.csv"))
        table.to_csv(out, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {out} ({len(table)} rows)")
    pd.set_option("display.width", 200)
    view = summary.pivot_table(index=["task", "condition", "regime"], columns="method", values=["macro_f1", "nll"])
    print(view.round(3).to_string())
    d = summary[summary.method == "dac"][["task", "condition", "regime", "phi_median", "phi_at_upper_bound_share",
                                          "phi_zero_share", "mixing_weight_target_mean",
                                          "dac_nll_win_share", "dac_macro_f1_change_max_abs", "dac_auroc_change_mean"]]
    print(d.round(4).to_string(index=False))
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
