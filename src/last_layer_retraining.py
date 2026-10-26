"""Few-label last-layer retraining of the fine-tuned encoders on frozen [CLS] features under shift.

For every converged cross-entropy base-grid run the [CLS] hidden states of the last layer (saved by
src/train.py, before the classification head) are standardised with the source calibration split and
projected on its first N_COMPONENTS principal components (label-free). Linear softmax heads on these features:
  source_linear_head   fitted on the labelled source calibration split (no target labels)
  retrained_head       fitted on n labelled target calibration items with an L2 penalty towards the source
                       head (MAP under a Gaussian prior centred at the source head)
  scratch_head         fitted on the n target items with an L2 penalty towards zero
The penalty strength is chosen from LAMBDAS by stratified cross-validation within the labelled items only
(source calibration split for the source head; the n target items otherwise). n, resamples and target
calibration subsets are those of src/shift_recalibration.py; metrics are computed on the LLM item subset of
each target test split, so rows are comparable with llm_recalibration_per_system.csv,
batch_calibration_per_system.csv and selective_prediction_per_system.csv.

Requires the hidden states of python -m src.fetch_run_outputs --hidden-target-splits.

Tables (results/tables/grid_max_length_512/):
  last_layer_retraining_per_run.csv   per run, condition, method and n (mean over resamples)
  last_layer_retraining_summary.csv   per encoder model and over all encoders

    python -m src.last_layer_retraining                 (--workers sets the number of processes)
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from sklearn.model_selection import StratifiedKFold

from src.config import TASKS
from src.llm_evaluation import GRID_DIR, TABLES, encoder_runs, llm_sets, split_labels
from src.shift_aware_routing import target_conditions
from src.shift_recalibration import metrics, subset_indices

N_COMPONENTS = 128
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0)
N_LABELS = (10, 25, 50, 100)
N_RESAMPLES = 20
OUTCOMES = ("accuracy", "macro_f1", "nll", "ece_15", "brier", "aurc", "auroc")


def fit_head(x, y, k, lam, w0=None) -> np.ndarray:
    """Softmax regression minimising mean cross-entropy + lam / 2 * ||W - W0||^2 (bias unpenalised).
    Returns the (d + 1) x k parameter matrix with the bias in the last row."""
    d = x.shape[1]
    w0 = np.zeros((d + 1, k)) if w0 is None else w0
    xb = np.hstack([x, np.ones((len(x), 1))])
    onehot = np.eye(k)[y]
    mask = np.ones((d + 1, k))
    mask[-1] = 0.0

    def objective(theta):
        w = theta.reshape(d + 1, k)
        logp = log_softmax(xb @ w, axis=1)
        diff = (w - w0) * mask
        value = -np.mean(np.sum(onehot * logp, axis=1)) + 0.5 * lam * np.sum(diff ** 2)
        grad = xb.T @ (np.exp(logp) - onehot) / len(x) + lam * diff
        return value, grad.ravel()

    return minimize(objective, w0.ravel(), jac=True, method="L-BFGS-B").x.reshape(d + 1, k)


def predict(w, x) -> np.ndarray:
    return softmax(np.hstack([x, np.ones((len(x), 1))]) @ w, axis=1)


def select_lambda(x, y, k, w0=None) -> float:
    counts = np.bincount(y, minlength=k)
    folds = int(min(3, counts[counts > 0].min()))
    if folds < 2 or (counts > 0).sum() < 2:
        return 1e-1
    losses = []
    for lam in LAMBDAS:
        nll = []
        for tr, va in StratifiedKFold(folds, shuffle=True, random_state=0).split(x, y):
            p = predict(fit_head(x[tr], y[tr], k, lam, w0), x[va])
            nll.append(-np.mean(np.log(np.clip(p[np.arange(len(va)), y[va]], 1e-12, 1.0))))
        losses.append(np.mean(nll))
    return LAMBDAS[int(np.argmin(losses))]


class FeatureSpace:
    """Standardisation and PCA fitted on the unlabeled source calibration features."""

    def __init__(self, x_source: np.ndarray, n_components: int):
        self.mean, self.std = x_source.mean(axis=0), x_source.std(axis=0) + 1e-6
        z = (x_source - self.mean) / self.std
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        self.components = vt[:n_components].T
        self.scale = (z @ self.components).std(axis=0) + 1e-6

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std) @ self.components / self.scale


def load_hidden(run_dir, split: str) -> np.ndarray:
    return np.load(run_dir / f"{split}_hidden.npy").astype(np.float64)


def process_run(run: dict, items_by: dict, gold: dict, n_resamples: int) -> list:
    d, task = run["dir"], run["task"]
    k = TASKS[task]
    if not (d / "cal_hidden.npy").exists():
        return []
    y_src = gold[task]["cal"]
    h_src = load_hidden(d, "cal")
    if len(h_src) != len(y_src):
        raise ValueError(f"{d.name}: cal hidden states do not match the labels")
    space = FeatureSpace(h_src, N_COMPONENTS)
    x_src = space(h_src)
    w_src = fit_head(x_src, y_src, k, select_lambda(x_src, y_src, k))
    key = {"run_id": d.name, "task": task, "model": run["model"],
           "ft_method": d.name.split(f"{run['model']}_", 1)[1].split("_seed")[0],
           "seed": int(d.name.rsplit("seed", 1)[1])}
    rows = []
    for cond in ["test"] + target_conditions(task):
        items = items_by.get((task, cond))
        if items is None or not (d / f"{cond}_hidden.npy").exists():
            continue
        y_t = gold[task][cond][items]
        x_t = space(load_hidden(d, cond)[items])
        base = {**key, "condition": cond, "n_items": len(items)}
        rows.append({**base, "method": "source_linear_head", "n_labels": "0", "resample": -1, "lambda": np.nan,
                     **metrics(predict(w_src, x_t), y_t)})
        if cond == "test":
            continue
        cal = cond[: -len("_test")] + "_cal"
        y_c = gold[task][cal]
        x_c = space(load_hidden(d, cal))
        sizes = [(n, r, subset_indices(len(y_c), n, r)) for n in N_LABELS if n < len(y_c)
                 for r in range(n_resamples)] + [("all", 0, np.arange(len(y_c)))]
        for n, r, sel in sizes:
            xs, ys = x_c[sel], y_c[sel]
            for method, w0 in (("retrained_head", w_src), ("scratch_head", None)):
                lam = select_lambda(xs, ys, k, w0)
                w = fit_head(xs, ys, k, lam, w0)
                rows.append({**base, "method": method, "n_labels": str(n), "resample": r, "lambda": lam,
                             **metrics(predict(w, x_t), y_t)})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Few-label last-layer retraining on frozen encoder features.")
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()
    start = time.time()
    # One BLAS thread per worker process; the workers inherit this environment.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"

    gold = {task: split_labels(task) for task in TASKS}
    sets = llm_sets(GRID_DIR / "llm_runs", gold)
    items_by = {}
    for (model, setting, task, split), v in sets.items():
        idx = np.sort(v["subset"]["indices"])
        if (task, split) in items_by and not np.array_equal(items_by[(task, split)], idx):
            raise ValueError(f"{task} {split}: LLM item subsets differ between systems")
        items_by[(task, split)] = idx
    runs = encoder_runs()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_run, run, items_by, gold, args.n_resamples): run["dir"].name for run in runs}
        for i, future in enumerate(as_completed(futures)):
            rows += future.result()
            print(f"[{i + 1}/{len(runs)}] {futures[future]} ({time.time() - start:.0f} s)", flush=True)
    print(f"{args.workers} worker processes")

    per = pd.DataFrame(rows)
    keys = ["run_id", "task", "model", "ft_method", "seed", "condition", "n_items", "method", "n_labels"]
    per_run = per.groupby(keys).agg(**{c: (c, "mean") for c in OUTCOMES}, lambda_median=("lambda", "median"),
                                    nll_resample_sd=("nll", "std")).reset_index()
    pooled = pd.concat([per_run, per_run.assign(model="all_encoders")], ignore_index=True)
    skeys = ["task", "condition", "model", "method", "n_labels"]
    summary = pooled.groupby(skeys)[list(OUTCOMES)].mean()
    summary["n_runs"] = pooled.groupby(skeys).run_id.size()
    summary = summary.reset_index()
    order = lambda df: df.assign(_n=df.n_labels.map(lambda v: float("inf") if v == "all" else float(v))).sort_values(
        [c for c in ("task", "condition", "model", "run_id", "method") if c in df.columns] + ["_n"]).drop(columns="_n")
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, table in [("last_layer_retraining_per_run.csv", order(per_run)),
                        ("last_layer_retraining_summary.csv", order(summary))]:
        table.to_csv(TABLES / name, index=False, float_format="%.6g", lineterminator="\n")
        print(f"Wrote {TABLES / name} ({len(table)} rows)")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    view = summary[summary.model == "all_encoders"].pivot_table(index=["task", "condition", "method"],
                                                                 columns="n_labels", values="macro_f1")
    view = view[sorted(view.columns, key=lambda c: float("inf") if c == "all" else float(c))]
    print("\nMean macro-F1 over encoder runs (n_labels 0 = no target labels)")
    print(view.round(3).to_string())
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
