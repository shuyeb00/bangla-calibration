"""Release bundle of the saved model outputs (level 2 of REPRODUCIBILITY.md) and its restoration.

The bundle holds the logits and item indices of every converged encoder run and every LLM system of the
max_length-512 study, as deterministic xz-compressed tar shards with a SHA-256 manifest. It contains no dataset text
and no labels: label arrays are removed and their SHA-256 is recorded, and `restore` rebuilds them from the processed
splits written by the preparation scripts (data/DATA_SOURCES.md), checking every array against the recorded hash.

Families (relative to results/grid_max_length_512/):
  runs/<run_id>/                    encoder logits per split (training-split logits and hidden states are not used by
                                    any analysis and are left out), configuration and completion records
  llm_runs/, llm_prompt_variants/,  LLM label log-probabilities, item indices and label mass per split; demonstration
  llm_target_demonstrations/        texts are replaced by their row numbers and the rendered prompt template is dropped
  few_label_finetuning/<run_id>/    logits of the fine-tuned encoders and the labelled rows used
  llm_membership_inference/<model>/ likelihood statistics; completion records keep scores and row numbers only

    python -m src.export_release_outputs export [--out release/outputs] [--anonymise]
    python -m src.export_release_outputs verify --bundle release/outputs
    python -m src.export_release_outputs restore --bundle release/outputs [--target DIR]
"""

import argparse
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (BASE_DIR, DATA_DIR, RESULTS_DIR, get_cross_definition_paths, get_cross_language_paths,
                        get_natural_codemixed_paths, get_shift_conditions_paths)

GRID_DIR = RESULTS_DIR / "grid_max_length_512"
TRAINING_SUMMARY = RESULTS_DIR / "tables" / "grid_max_length_512_training_summary.csv"
DEFAULT_OUT = BASE_DIR / "release" / "outputs"
LLM_FAMILIES = ("llm_runs", "llm_prompt_variants", "llm_target_demonstrations")
FAMILIES = ("runs",) + LLM_FAMILIES + ("few_label_finetuning", "llm_membership_inference")
FORMAT_VERSION = 1
XZ_PRESET = 6
LLM_CONFIG_DROP = ("prompt_template",)
COMPLETION_KEEP = ("file", "row")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_digest(a: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(a).tobytes())


def npy_bytes(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, a, allow_pickle=False)
    return buf.getvalue()


def npz_bytes(arrays: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def label_record(path: str, a: np.ndarray, task: str, split: str, rows) -> dict:
    return {"path": path, "dtype": str(a.dtype), "shape": list(a.shape), "sha256": array_digest(a), "task": task,
            "split": split, "rows": rows}


def run_task(cfg: dict, run_dir: Path) -> str:
    return cfg.get("task") or run_dir.name.split("_")[0]


def encoder_run(run_dir: Path, include_train: bool) -> tuple:
    """Files and label records of one encoder run directory."""
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    task, rel = run_task(cfg, run_dir), f"runs/{run_dir.name}"
    files, labels, skipped = {}, [], {"hidden_states": 0, "train_split": 0}
    for p in sorted(run_dir.iterdir()):
        if p.suffix == ".json":
            files[f"{rel}/{p.name}"] = p.read_bytes()
        elif p.name.endswith("_hidden.npy"):
            skipped["hidden_states"] += 1
        elif p.name.endswith("_labels.npy") or p.name.endswith("_logits.npy"):
            split = p.name.rsplit("_", 1)[0]
            if split == "train" and not include_train:
                skipped["train_split"] += 1
                continue
            if p.name.endswith("_logits.npy"):
                files[f"{rel}/{p.name}"] = p.read_bytes()
            else:
                labels.append(label_record(f"{rel}/{p.name}", np.load(p), task, split, None))
    return task, files, labels, skipped


def sanitized_llm_config(cfg: dict) -> dict:
    out = {k: v for k, v in cfg.items() if k not in LLM_CONFIG_DROP}
    if out.get("demonstrations"):
        out["demonstrations"] = [{"train_row": d["train_row"]} for d in out["demonstrations"]]
    out["release_note"] = "demonstration texts, their labels and the rendered prompt template are removed"
    return out


def llm_system(family: str, run_dir: Path) -> tuple:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    task, rel = cfg["task"], f"{family}/{run_dir.name}"
    files, labels = {f"{rel}/config.json": json_bytes(sanitized_llm_config(cfg))}, []
    for p in sorted(run_dir.iterdir()):
        if p.name == "config.json":
            continue
        if p.name.endswith("_demonstrations.json"):
            demos = json.loads(p.read_text(encoding="utf-8"))
            files[f"{rel}/{p.name}"] = json_bytes([{"cal_split": d["cal_split"], "cal_row": d["cal_row"]}
                                                   for d in demos])
        elif p.suffix == ".json":
            files[f"{rel}/{p.name}"] = p.read_bytes()
        elif p.name.endswith("_labels.npy"):
            part = p.name[: -len("_labels.npy")]
            labels.append(label_record(f"{rel}/{p.name}", np.load(p), task, part.replace("_remainder", ""),
                                       f"{rel}/{part}_indices.npy"))
        elif p.suffix == ".npy":
            files[f"{rel}/{p.name}"] = p.read_bytes()
    return task, files, labels


def few_label_run(run_dir: Path) -> tuple:
    rel = f"few_label_finetuning/{run_dir.name}"
    task = next(t for t in ("offensive", "sentiment") if run_dir.name.startswith(f"{t}_"))
    files, labels = {}, []
    for p in sorted(run_dir.iterdir()):
        if p.suffix == ".json":
            files[f"{rel}/{p.name}"] = p.read_bytes()
        elif p.suffix == ".npz":
            d = dict(np.load(p))
            keep = {}
            for key, a in d.items():
                if key == "test_labels":
                    labels.append(label_record(f"{rel}/{p.name}::{key}", a, task, f"{p.stem}_test",
                                               f"{rel}/{p.name}::test_rows"))
                elif key.endswith("_val_labels"):
                    labels.append(label_record(f"{rel}/{p.name}::{key}", a, task, f"{p.stem}_cal",
                                               f"{rel}/{p.name}::{key[: -len('_labels')]}_rows"))
                else:
                    keep[key] = a
            files[f"{rel}/{p.name}"] = npz_bytes(keep)
    return task, files, labels


def membership_model(model_dir: Path) -> dict:
    rel, files = f"llm_membership_inference/{model_dir.name}", {}
    for p in sorted(model_dir.iterdir()):
        if p.name.endswith("_completion.jsonl"):
            items = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
            kept = [{k: v for k, v in x.items() if k in COMPLETION_KEEP or isinstance(v, (bool, int, float))}
                    for x in items]
            files[f"{rel}/{p.name}"] = ("\n".join(json.dumps(x, sort_keys=True) for x in kept) + "\n").encode()
        elif p.suffix in (".npz", ".json"):
            files[f"{rel}/{p.name}"] = p.read_bytes()
    return files


def non_converged() -> set:
    s = pd.read_csv(TRAINING_SUMMARY)
    return set(s.loc[s["non_converged"].astype(bool), "run_id"])


def anonymised_json(data: bytes) -> bytes:
    """Identifying strings and code commits removed, for the double-blind supplementary archive."""
    from src.build_supplementary_archive import anonymise

    def redact(x):
        if isinstance(x, dict):
            return {k: "redacted" if "commit" in k.lower() else redact(v) for k, v in x.items()}
        return [redact(v) for v in x] if isinstance(x, list) else x

    return anonymise("file.json", json_bytes(redact(json.loads(data))).decode("utf-8")).encode("utf-8")


def collect(grid_dir: Path, include_train: bool) -> tuple:
    """Shard name -> {path in bundle: bytes}, label records, notes on what was left out."""
    shards, labels = {}, []
    notes = {"hidden_states": 0, "train_split": 0, "non_converged_runs": []}
    excluded = non_converged()
    for run_dir in sorted(p for p in (grid_dir / "runs").glob("*") if (p / "config.json").exists()):
        if run_dir.name in excluded:
            notes["non_converged_runs"].append(run_dir.name)
            continue
        task, files, recs, skipped = encoder_run(run_dir, include_train)
        shards.setdefault(f"runs_{task}", {}).update(files)
        labels += recs
        for k, v in skipped.items():
            notes[k] += v
    for family in LLM_FAMILIES:
        for run_dir in sorted(p for p in (grid_dir / family).glob("*") if (p / "config.json").exists()):
            task, files, recs = llm_system(family, run_dir)
            shards.setdefault(f"{family}_{task}", {}).update(files)
            labels += recs
    for run_dir in sorted(p for p in (grid_dir / "few_label_finetuning").glob("*") if p.is_dir()):
        _, files, recs = few_label_run(run_dir)
        shards.setdefault("few_label_finetuning", {}).update(files)
        labels += recs
    for model_dir in sorted(p for p in (grid_dir / "llm_membership_inference").glob("*") if p.is_dir()):
        shards.setdefault("llm_membership_inference", {}).update(membership_model(model_dir))
    return shards, labels, notes


def write_shard(path: Path, files: dict) -> None:
    with tarfile.open(path, "w:xz", preset=XZ_PRESET, format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(files[name]), 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(files[name]))


def export(out: Path, include_train: bool, grid_dir: Path = GRID_DIR, anonymised: bool = False) -> dict:
    t0 = time.time()
    shards, labels, notes = collect(grid_dir, include_train)
    if anonymised:
        shards = {s: {f: anonymised_json(d) if f.endswith(".json") else d for f, d in files.items()}
                  for s, files in shards.items()}
    if not shards:
        raise SystemExit(f"No outputs found under {grid_dir}; fetch them first (REPRODUCIBILITY.md, level 2).")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"format_version": FORMAT_VERSION, "grid": "max_length_512", "anonymised": anonymised,
                "shards": {}, "files": {},
                "labels": sorted(labels, key=lambda r: r["path"]), "left_out": notes,
                "restore": "python -m src.export_release_outputs restore --bundle <this directory>"}
    for name in sorted(shards):
        path = out / f"{name}.tar.xz"
        write_shard(path, shards[name])
        manifest["shards"][path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size,
                                         "files": len(shards[name])}
        for f, data in shards[name].items():
            manifest["files"][f] = {"shard": path.name, "sha256": sha256_bytes(data), "bytes": len(data)}
        print(f"{path.name}: {len(shards[name])} files, {path.stat().st_size / 2**20:.1f} MB")
    (out / "MANIFEST.json").write_bytes(json_bytes(manifest))
    lines = [f"{v['sha256']}  {k}" for k, v in sorted(manifest["shards"].items())]
    lines.append(f"{sha256_file(out / 'MANIFEST.json')}  MANIFEST.json")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = sum(v["bytes"] for v in manifest["shards"].values())
    print(f"{len(manifest['files'])} files in {len(manifest['shards'])} shards, {total / 2**20:.1f} MB; "
          f"{len(labels)} label arrays removed; left out: {notes['hidden_states']} hidden-state files, "
          f"{notes['train_split']} training-split files, non-converged runs {notes['non_converged_runs']}; "
          f"{time.time() - t0:.0f} s")
    return manifest


def verify(bundle: Path) -> dict:
    manifest = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))
    sums = dict(line.split("  ")[::-1] for line in (bundle / "SHA256SUMS").read_text().splitlines() if line)
    bad = [n for n, v in manifest["shards"].items() if sha256_file(bundle / n) != v["sha256"]]
    if sums.get("MANIFEST.json") != sha256_file(bundle / "MANIFEST.json"):
        bad.append("MANIFEST.json")
    if bad:
        raise SystemExit(f"Checksum mismatch: {bad}")
    print(f"{len(manifest['shards'])} shards and the manifest match their SHA-256 sums")
    return manifest


def split_paths(task: str) -> dict:
    paths = {s: DATA_DIR / task / f"{s}.csv" for s in ("train", "val", "cal", "test")}
    for fn in (get_shift_conditions_paths, get_natural_codemixed_paths, get_cross_definition_paths,
               get_cross_language_paths):
        paths.update(fn(task))
    return paths


def split_labels(task: str, split: str, cache: dict) -> np.ndarray:
    key = (task, split)
    if key not in cache:
        path = split_paths(task).get(split)
        if path is None or not Path(path).exists():
            raise FileNotFoundError(f"{task}/{split}: processed split missing; run its preparation script "
                                    f"(data/DATA_SOURCES.md)")
        df = pd.read_csv(path)
        cache[key] = df["labels" if "labels" in df.columns else "label"].to_numpy()
    return cache[key]


def restore(bundle: Path, target: Path, overwrite: bool) -> None:
    t0 = time.time()
    manifest = verify(bundle)
    target.mkdir(parents=True, exist_ok=True)
    written = 0
    for name in sorted(manifest["shards"]):
        with tarfile.open(bundle / name, "r:xz") as tar:
            for member in tar.getmembers():
                dest = (target / member.name).resolve()
                if not member.isfile() or target.resolve() not in dest.parents:
                    raise SystemExit(f"Unexpected member {member.name} in {name}")
                data = tar.extractfile(member).read()
                if sha256_bytes(data) != manifest["files"][member.name]["sha256"]:
                    raise SystemExit(f"Checksum mismatch for {member.name}")
                if dest.exists() and not overwrite:
                    raise SystemExit(f"{dest} exists; use --overwrite or another --target")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                written += 1
    cache, npz_pending, missing, mismatched, rebuilt = {}, {}, set(), [], 0
    for rec in manifest["labels"]:
        try:
            y = split_labels(rec["task"], rec["split"], cache)
        except FileNotFoundError as e:
            missing.add(str(e))
            continue
        rows = rec["rows"]
        if rows is not None:
            if "::" in rows:
                f, k = rows.split("::")
                rows = np.load(target / f)[k]
            else:
                rows = np.load(target / rows)
            y = y[rows]
        y = np.ascontiguousarray(y.astype(rec["dtype"]).reshape(rec["shape"]))
        if array_digest(y) != rec["sha256"]:
            mismatched.append(rec["path"])
            continue
        rebuilt += 1
        if "::" in rec["path"]:
            f, k = rec["path"].split("::")
            npz_pending.setdefault(f, {})[k] = y
        else:
            np.save(target / rec["path"], y)
    for f, arrays in npz_pending.items():
        with np.load(target / f) as d:
            merged = {**dict(d), **arrays}
        np.savez_compressed(target / f, **merged)
    print(f"Restored {written} files and rebuilt {rebuilt} of {len(manifest['labels'])} label arrays into {target} "
          f"in {time.time() - t0:.0f} s")
    if missing:
        print("Missing processed splits:\n  " + "\n  ".join(sorted(missing)))
    if mismatched:
        raise SystemExit(f"{len(mismatched)} rebuilt label arrays differ from the released hashes, e.g. "
                         f"{mismatched[:3]}; check the preparation scripts' checksums (data/DATA_SOURCES.md).")
    if missing:
        raise SystemExit(f"{len(manifest['labels']) - rebuilt} label arrays not rebuilt; prepare the missing splits "
                         f"and run restore again with --overwrite.")


def main():
    parser = argparse.ArgumentParser(description="Export, verify or restore the released model outputs.")
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", type=Path, default=DEFAULT_OUT)
    e.add_argument("--grid-dir", type=Path, default=GRID_DIR)
    e.add_argument("--include-train", action="store_true", help="also bundle training-split logits (unused)")
    e.add_argument("--anonymise", action="store_true", help="redact identifying strings for double-blind review")
    v = sub.add_parser("verify")
    v.add_argument("--bundle", type=Path, default=DEFAULT_OUT)
    r = sub.add_parser("restore")
    r.add_argument("--bundle", type=Path, default=DEFAULT_OUT)
    r.add_argument("--target", type=Path, default=GRID_DIR)
    r.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "export":
        export(args.out, args.include_train, args.grid_dir, args.anonymise)
    elif args.command == "verify":
        verify(args.bundle)
    else:
        restore(args.bundle, args.target, args.overwrite)


if __name__ == "__main__":
    main()
