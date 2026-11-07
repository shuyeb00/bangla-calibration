"""Machine translation of the two offensive-language source corpora for the definition-by-language factorial.

English OLID (src/prepare_olid.py) is translated into Bangla and BD-SHS into English with NLLB-200 3.3B (NLLB Team,
2022; CC-BY-NC-4.0, research use), and every forward translation is translated back into the source language. With
the original corpora this gives each definition in both languages, with zero, one or two machine translation passes
(config.SOURCE_FACTORS). Labels are those of the original items; no item is filtered by translation quality.

The text that is translated is the cleaned text the original encoders were trained on (text_clean). U+FFFD replacement
characters of BD-SHS (lost symbols, progress doc section 3) are removed first. Decoding: beam search (MT_SETTINGS), at
most MAX_INPUT_TOKENS source tokens, at most 1.5 times the source length plus 10 new tokens.

Steps
  translate  shard k of N of all items (both corpora, splits train/val/cal/test; item i belongs to shard i mod N):
             forward translation, back translation of the forward output, and LaBSE cosine similarity (Feng et al.,
             2022) of source and forward translation and of source and back translation; one worker process per GPU.
             Writes machine_translation/shard{k}_of_{N}.csv.gz and its completion record to the store. With --wait,
             the shards of other workers that are still missing after STALE_CLAIM_MINUTES are taken over, and the
             command returns when all N shards are in the store.
  assemble   downloads all N shards, checks that every item is present exactly once and that the source texts and
             labels equal the prepared corpora, cleans the outputs like the original data (Bangla: the csebuetnlp
             normaliser as src/prepare_natural_codemixed.py; English: URL removal, whitespace, NFKC as
             src/prepare_olid.py) and writes data/processed/{task}/{train,val,cal,test}.csv for the four tasks of
             config.TRANSLATION_FACTORIAL_TASKS (rows with an empty cleaned translation are dropped and counted),
             data/processed/translation_preparation.json and the text-free quality table translation_quality.csv
             (tables folder of the max_length-512 grid).

Quality is measured automatically only: LaBSE similarities, round-trip chrF++ and BLEU (sacrebleu; Popovic, 2017;
Post, 2018), the share of letters in the target script, copies of the source, degenerate repetition, length ratios,
retention of the @USER placeholder, all per corpus, split and label (a translation that removes offensive cues would
show a lower similarity for offensive items).

    python -m src.translate_source_corpora translate --shard 0 --num-shards 7 --store hub --wait
    python -m src.translate_source_corpora assemble --num-shards 7 --store hub
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR, GRID_LAYOUTS, HUB_OUTPUT_REPO

MT_MODEL = "facebook/nllb-200-3.3B"
LABSE_MODEL = "setu4993/LaBSE"
LANG_CODES = {"english": "eng_Latn", "bangla": "ben_Beng"}
# corpus -> (prepared task folder, source language, other language)
CORPORA = {"olid": ("offensive_olid", "english", "bangla"), "bdshs": ("offensive", "bangla", "english")}
SPLITS = ["train", "val", "cal", "test"]
# translated task -> (corpus, pass whose output is the training text)
TASK_SOURCES = {
    "offensive_olid_bangla_mt": ("olid", "forward"),
    "offensive_olid_english_roundtrip": ("olid", "backward"),
    "offensive_bdshs_english_mt": ("bdshs", "forward"),
    "offensive_bdshs_bangla_roundtrip": ("bdshs", "backward"),
}
MT_SETTINGS = {"num_beams": 4, "max_input_tokens": 256, "max_new_tokens_factor": 1.5, "max_new_tokens_offset": 10}
MAX_BATCH_TOKENS = 3072
MAX_BATCH_ITEMS = 96
STORE_PREFIX = "machine_translation"
STALE_CLAIM_MINUTES = float(os.environ.get("TRANSLATION_STALE_CLAIM_MINUTES", 120))
POLL_SECONDS = float(os.environ.get("TRANSLATION_POLL_SECONDS", 120))
WORK_DIR = Path(os.environ.get("TRANSLATION_WORK_DIR", "/tmp/translation_work"))
QUALITY_TABLE = GRID_LAYOUTS["max_length_512"][1] / "translation_quality.csv"
PREPARATION_RECORD = DATA_DIR / "translation_preparation.json"
SCRIPT_RANGES = {"bangla": ((0x0980, 0x09FF),), "english": ((0x0041, 0x005A), (0x0061, 0x007A))}


def shard_name(k: int, n: int) -> str:
    return f"shard{k:02d}_of_{n:02d}"


class LocalStore:
    def __init__(self, root: str):
        self.root = Path(root) / STORE_PREFIX
        self.root.mkdir(parents=True, exist_ok=True)

    def exists(self, name: str) -> bool:
        return (self.root / name).exists()

    def read_json(self, name: str):
        path = self.root / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def put(self, local: Path, name: str) -> None:
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, target)

    def put_json(self, payload: dict, name: str) -> None:
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get(self, name: str, local_dir: Path) -> Path:
        local_dir.mkdir(parents=True, exist_ok=True)
        return Path(shutil.copy(self.root / name, local_dir / Path(name).name))


class HubStore:
    def __init__(self, repo: str = HUB_OUTPUT_REPO):
        from huggingface_hub import HfApi

        self.repo = repo
        self.api = HfApi(token=os.environ.get("HF_TOKEN"))

    def _retry(self, action, what: str, attempts: int = 6):
        for attempt in range(attempts):
            try:
                return action()
            except Exception as exc:
                if attempt == attempts - 1:
                    raise
                wait = 30 * (attempt + 1)
                print(f"  [STORE RETRY] {what}: {type(exc).__name__}: {str(exc)[:200]}; retry in {wait} s", flush=True)
                time.sleep(wait)

    def exists(self, name: str) -> bool:
        return self._retry(lambda: self.api.file_exists(self.repo, f"{STORE_PREFIX}/{name}", repo_type="dataset"),
                           f"exists {name}")

    def read_json(self, name: str):
        if not self.exists(name):
            return None
        path = self.get(name, WORK_DIR / "store_json")
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def put(self, local: Path, name: str) -> None:
        self._retry(lambda: self.api.upload_file(path_or_fileobj=str(local), path_in_repo=f"{STORE_PREFIX}/{name}",
                                                 repo_id=self.repo, repo_type="dataset",
                                                 commit_message=f"Add {STORE_PREFIX}/{name}"), f"upload {name}")

    def put_json(self, payload: dict, name: str) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self._retry(lambda: self.api.upload_file(path_or_fileobj=io.BytesIO(data), path_in_repo=f"{STORE_PREFIX}/{name}",
                                                 repo_id=self.repo, repo_type="dataset",
                                                 commit_message=f"Add {STORE_PREFIX}/{name}"), f"upload {name}")

    def get(self, name: str, local_dir: Path) -> Path:
        from huggingface_hub import hf_hub_download

        local_dir.mkdir(parents=True, exist_ok=True)
        return Path(self._retry(lambda: hf_hub_download(self.repo, f"{STORE_PREFIX}/{name}", repo_type="dataset",
                                                        token=os.environ.get("HF_TOKEN"), local_dir=str(local_dir)),
                                f"download {name}"))


def make_store(spec: str):
    if spec == "hub":
        return HubStore()
    if spec.startswith("local:"):
        return LocalStore(spec[len("local:"):])
    raise ValueError(f"--store must be hub or local:<dir>, got {spec!r}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_text(corpus: str, text: str) -> str:
    text = str(text)
    if corpus == "bdshs":
        text = re.sub(r"\s+", " ", text.replace("�", " ")).strip()
    return text


def all_items(limit_rows: int = None) -> pd.DataFrame:
    """Every item of both corpora in a fixed order; item_index is the position in that order."""
    frames = []
    for corpus, (folder, src, other) in CORPORA.items():
        for split in SPLITS:
            path = DATA_DIR / folder / f"{split}.csv"
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing; prepare the source corpora first")
            df = pd.read_csv(path)
            if limit_rows:
                df = df.iloc[:limit_rows]
            frames.append(pd.DataFrame({
                "corpus": corpus, "split": split, "row": np.arange(len(df)), "labels": df["labels"].astype(int),
                "source_text": [source_text(corpus, t) for t in df["text_clean"].astype(str)],
                "source_lang": src, "target_lang": other}))
    items = pd.concat(frames, ignore_index=True)
    items.insert(0, "item_index", np.arange(len(items)))
    return items


def load_mt(model_name: str, device: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    return tokenizer, model


def encode_source(tokenizer, texts: list, lang: str, max_tokens: int) -> list:
    tokenizer.src_lang = LANG_CODES[lang]
    ids = [tokenizer(t, truncation=True, max_length=max_tokens)["input_ids"] for t in texts]
    lang_id = tokenizer.convert_tokens_to_ids(LANG_CODES[lang])
    bad = [i for i, x in enumerate(ids) if not x or x[0] != lang_id]
    if bad:
        raise RuntimeError(f"Source language token {LANG_CODES[lang]} missing in {len(bad)} encodings; "
                           f"tokenizer convention changed")
    return ids


def batches(lengths: list, max_tokens: int, max_items: int) -> list:
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    out, current, longest = [], [], 0
    for i in order:
        longest_next = max(longest, lengths[i])
        if current and (longest_next * (len(current) + 1) > max_tokens or len(current) >= max_items):
            out.append(current)
            current, longest_next = [], lengths[i]
        current.append(i)
        longest = longest_next
    if current:
        out.append(current)
    return out


def translate(texts: list, tokenizer, model, src: str, tgt: str, device: str, label: str) -> tuple:
    import torch

    ids = encode_source(tokenizer, texts, src, MT_SETTINGS["max_input_tokens"])
    lengths = [len(x) for x in ids]
    forced = tokenizer.convert_tokens_to_ids(LANG_CODES[tgt])
    out = [""] * len(texts)
    pending = batches(lengths, MAX_BATCH_TOKENS, MAX_BATCH_ITEMS)
    t0, done = time.time(), 0
    while pending:
        batch = pending.pop(0)
        enc = tokenizer.pad({"input_ids": [ids[i] for i in batch]}, return_tensors="pt").to(device)
        max_len = max(lengths[i] for i in batch)
        try:
            with torch.inference_mode():
                gen = model.generate(**enc, forced_bos_token_id=forced, num_beams=MT_SETTINGS["num_beams"],
                                     max_new_tokens=int(MT_SETTINGS["max_new_tokens_factor"] * max_len)
                                     + MT_SETTINGS["max_new_tokens_offset"])
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(batch) // 2
            pending[:0] = [batch[:half], batch[half:]]
            print(f"  [{label}] out of memory at {len(batch)} items; split the batch", flush=True)
            continue
        for i, text in zip(batch, tokenizer.batch_decode(gen, skip_special_tokens=True)):
            out[i] = text.strip()
        done += len(batch)
        if done // 2000 != (done - len(batch)) // 2000 or not pending:
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  [{label}] {src}->{tgt}: {done}/{len(texts)} items, {rate:.1f} items/s", flush=True)
    truncated = [n >= MT_SETTINGS["max_input_tokens"] for n in lengths]
    return out, truncated, time.time() - t0


def labse_similarities(model_name: str, source: list, others: list, device: str) -> list:
    """Cosine similarity of every source text with the corresponding text of each list in others."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(model_name, dtype=dtype).to(device).eval()

    def embed(texts):
        vecs = []
        for start in range(0, len(texts), 128):
            enc = tokenizer(texts[start:start + 128], padding=True, truncation=True, max_length=128,
                            return_tensors="pt").to(device)
            with torch.inference_mode():
                v = model(**enc).pooler_output.float()
            vecs.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return np.concatenate(vecs) if vecs else np.zeros((0, 1))

    reference = embed(source)
    sims = [(reference * embed(other)).sum(axis=1) for other in others]
    del model
    return sims


def run_worker(args) -> None:
    """Translate the items of one input file on one device."""
    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    items = pd.read_csv(args.input, keep_default_na=False)
    if args.load_after:
        while not Path(args.load_after).exists():
            time.sleep(5)
    tokenizer, model = load_mt(args.mt_model, device)
    if args.loaded_flag:
        Path(args.loaded_flag).write_text("loaded")
    label = Path(args.input).stem
    items["forward_text"], items["backward_text"] = "", ""
    items["forward_truncated"], items["backward_truncated"] = False, False
    seconds = {}
    for (src, tgt), idx in items.groupby(["source_lang", "target_lang"]).groups.items():
        rows = list(idx)
        fwd, trunc, sec = translate(items.loc[rows, "source_text"].tolist(), tokenizer, model, src, tgt, device,
                                    f"{label} forward")
        items.loc[rows, "forward_text"], items.loc[rows, "forward_truncated"] = fwd, trunc
        seconds[f"{src}->{tgt}"] = sec
        back, trunc, sec = translate(fwd, tokenizer, model, tgt, src, device, f"{label} backward")
        items.loc[rows, "backward_text"], items.loc[rows, "backward_truncated"] = back, trunc
        seconds[f"{tgt}->{src} (back)"] = sec
    del model
    if device == "cuda":
        __import__("torch").cuda.empty_cache()
    items["labse_forward"], items["labse_backward"] = labse_similarities(
        args.labse_model, items["source_text"].tolist(),
        [items["forward_text"].tolist(), items["backward_text"].tolist()], device)
    items.to_csv(args.output, index=False, encoding="utf-8", lineterminator="\n")
    Path(args.output).with_suffix(".json").write_text(json.dumps({"device": device, "seconds": seconds}),
                                                      encoding="utf-8")


def prefetch(name_or_path: str) -> None:
    """Download a Hub model once before the workers start (safetensors only when the repository has them)."""
    if Path(name_or_path).exists():
        return
    from huggingface_hub import HfApi, snapshot_download

    files = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(name_or_path)
    weights = ["*.safetensors"] if any(f.endswith(".safetensors") for f in files) else ["*.bin"]
    snapshot_download(name_or_path, allow_patterns=[*weights, "*.json", "*.model", "*.txt"],
                      token=os.environ.get("HF_TOKEN"))


def model_revision(name: str) -> str:
    try:
        from huggingface_hub import HfApi

        return HfApi(token=os.environ.get("HF_TOKEN")).model_info(name).sha
    except Exception:
        return "unknown"


def translate_shard(k: int, n: int, args, store) -> None:
    name = shard_name(k, n)
    if store.exists(f"{name}.json"):
        print(f"[SKIP] {name} already in the store", flush=True)
        return
    store.put_json({"claimed": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname()},
                   f"claims/{name}.json")
    items = all_items(args.limit_rows)
    shard = items[items["item_index"] % n == k].reset_index(drop=True)
    work = WORK_DIR / name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    devices = args.devices.split(",") if args.devices else [""]
    for name_or_path in (args.mt_model, args.labse_model):
        prefetch(name_or_path)
    procs = []
    t0 = time.time()
    for w, dev in enumerate(devices):
        part = shard.iloc[w::len(devices)]
        part.to_csv(work / f"part{w}.csv", index=False, encoding="utf-8", lineterminator="\n")
        env = dict(os.environ, TOKENIZERS_PARALLELISM="false")
        if dev != "":
            env["CUDA_VISIBLE_DEVICES"] = dev
        cmd = [sys.executable, "-u", "-m", "src.translate_source_corpora", "worker", "--input", str(work / f"part{w}.csv"),
               "--output", str(work / f"out{w}.csv"), "--mt-model", args.mt_model, "--labse-model", args.labse_model,
               "--loaded-flag", str(work / f"loaded{w}")]
        if w > 0:
            cmd += ["--load-after", str(work / f"loaded{w - 1}")]
        if args.device:
            cmd += ["--device", args.device]
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"[FAILED] {name}: worker return codes {codes}")
    out = pd.concat([pd.read_csv(work / f"out{w}.csv", keep_default_na=False) for w in range(len(devices))])
    out = out.sort_values("item_index").reset_index(drop=True)
    if len(out) != len(shard) or not (out["item_index"].to_numpy() == shard["item_index"].to_numpy()).all():
        raise SystemExit(f"[FAILED] {name}: worker outputs do not cover the shard")
    path = work / f"{name}.csv.gz"
    with gzip.GzipFile(path, "wb", mtime=0) as fh:
        fh.write(out.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    if store.exists(f"{name}.json"):
        print(f"[SKIP UPLOAD] {name} was completed elsewhere meanwhile", flush=True)
        return
    record = {"shard": k, "num_shards": n, "items": len(out), "sha256": sha256_file(path),
              "mt_model": args.mt_model, "mt_revision": model_revision(args.mt_model),
              "labse_model": args.labse_model, "labse_revision": model_revision(args.labse_model),
              "settings": {**MT_SETTINGS, "max_batch_tokens": MAX_BATCH_TOKENS, "max_batch_items": MAX_BATCH_ITEMS},
              "worker_seconds": [json.loads((work / f"out{w}.json").read_text()) for w in range(len(devices))],
              "wall_seconds": round(time.time() - t0, 1), "host": socket.gethostname(),
              "limit_rows": args.limit_rows, "completed": datetime.now(timezone.utc).isoformat()}
    store.put(path, f"{name}.csv.gz")
    store.put_json(record, f"{name}.json")
    print(f"[UPLOADED] {name}: {len(out)} items in {record['wall_seconds'] / 60:.1f} min", flush=True)


def claim_age_minutes(store, name: str) -> float:
    claim = store.read_json(f"claims/{name}.json")
    if not claim:
        return float("inf")
    claimed = datetime.fromisoformat(claim["claimed"])
    return (datetime.now(timezone.utc) - claimed).total_seconds() / 60


def run_translate(args) -> None:
    store = make_store(args.store)
    translate_shard(args.shard, args.num_shards, args, store)
    if not args.wait:
        return
    deadline = time.time() + args.wait_hours * 3600
    order = [(args.shard + j) % args.num_shards for j in range(1, args.num_shards)]
    while True:
        missing = [j for j in order if not store.exists(f"{shard_name(j, args.num_shards)}.json")]
        if not missing:
            print(f"[ALL SHARDS] {args.num_shards} of {args.num_shards} in the store", flush=True)
            return
        if time.time() > deadline:
            raise SystemExit(f"[TIMEOUT] shards still missing: {missing}")
        stale = [j for j in missing if claim_age_minutes(store, shard_name(j, args.num_shards)) > STALE_CLAIM_MINUTES]
        if stale:
            print(f"[TAKE OVER] shard {stale[0]} (no completion within {STALE_CLAIM_MINUTES} min of its claim)",
                  flush=True)
            translate_shard(stale[0], args.num_shards, args, store)
            continue
        print(f"[WAIT] shards missing: {missing}", flush=True)
        time.sleep(POLL_SECONDS)


def script_share(text: str, lang: str) -> float:
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return float("nan")
    hit = sum(any(lo <= ord(c) <= hi for lo, hi in SCRIPT_RANGES[lang]) for c in letters)
    return hit / len(letters)


def degenerate(text: str) -> bool:
    words = text.split()
    grams = Counter(tuple(words[i:i + 4]) for i in range(len(words) - 3))
    return bool(grams) and max(grams.values()) >= 3


def clean_output(text: str, lang: str) -> str:
    if lang == "bangla":
        from src.prepare_natural_codemixed import clean_text

        return clean_text(text)
    from src.prepare_olid import clean_text

    return clean_text(text)


def quality_rows(frame: pd.DataFrame) -> list:
    import sacrebleu

    rows = []
    groups = [(c, s, l) for c in CORPORA for s in [*SPLITS, "all"] for l in ("all", 0, 1)]
    for corpus, split, label in groups:
        d = frame[frame["corpus"] == corpus]
        if split != "all":
            d = d[d["split"] == split]
        if label != "all":
            d = d[d["labels"] == label]
        if d.empty:
            continue
        src, tgt = CORPORA[corpus][1], CORPORA[corpus][2]
        mentions = d["source_text"].str.contains("@user", case=False)
        roundtrip = sacrebleu.corpus_chrf(d["backward_text"].tolist(), [d["source_text"].tolist()], word_order=2)
        row = {"corpus": corpus, "split": split, "label": label, "items": len(d),
               "source_language": src, "target_language": tgt,
               "labse_forward_mean": d["labse_forward"].mean(), "labse_forward_median": d["labse_forward"].median(),
               "labse_forward_p05": d["labse_forward"].quantile(0.05),
               "labse_backward_mean": d["labse_backward"].mean(),
               "roundtrip_chrf_pp": roundtrip.score,
               "roundtrip_bleu": (sacrebleu.corpus_bleu(d["backward_text"].tolist(), [d["source_text"].tolist()]).score
                                  if src == "english" else np.nan),
               "forward_target_script_share": d["forward_text"].map(lambda t: script_share(t, tgt)).mean(),
               "backward_source_script_share": d["backward_text"].map(lambda t: script_share(t, src)).mean(),
               "forward_copy_rate": (d["forward_text"] == d["source_text"]).mean(),
               "forward_degenerate_rate": d["forward_text"].map(degenerate).mean(),
               "backward_degenerate_rate": d["backward_text"].map(degenerate).mean(),
               "forward_length_ratio_median": (d["forward_text"].str.len() / d["source_text"].str.len().clip(lower=1)).median(),
               "backward_length_ratio_median": (d["backward_text"].str.len() / d["source_text"].str.len().clip(lower=1)).median(),
               "forward_empty_rate": d["forward_clean"].eq("").mean(),
               "backward_empty_rate": d["backward_clean"].eq("").mean(),
               "input_truncated_rate": d["forward_truncated"].astype(str).eq("True").mean(),
               "user_placeholder_items": int(mentions.sum()),
               "user_placeholder_forward_kept": (d.loc[mentions, "forward_text"].str.contains("@user", case=False).mean()
                                                 if mentions.any() else np.nan),
               "user_placeholder_backward_kept": (d.loc[mentions, "backward_text"].str.contains("@user", case=False).mean()
                                                  if mentions.any() else np.nan)}
        rows.append(row)
    return rows


def run_assemble(args) -> None:
    store = make_store(args.store)
    n = args.num_shards
    deadline = time.time() + args.wait_hours * 3600
    while True:
        missing = [k for k in range(n) if not store.exists(f"{shard_name(k, n)}.json")]
        if not missing:
            break
        if time.time() > deadline:
            raise SystemExit(f"[TIMEOUT] shards still missing: {missing}")
        print(f"[WAIT] shards missing: {missing}", flush=True)
        time.sleep(POLL_SECONDS)
    local = WORK_DIR / "assemble"
    shutil.rmtree(local, ignore_errors=True)
    frames, records = [], []
    for k in range(n):
        record = store.read_json(f"{shard_name(k, n)}.json")
        path = store.get(f"{shard_name(k, n)}.csv.gz", local)
        if sha256_file(path) != record["sha256"]:
            raise SystemExit(f"[FAILED] checksum of {shard_name(k, n)} differs from its record")
        frames.append(pd.read_csv(path, keep_default_na=False))
        records.append(record)
    if len({r["mt_revision"] for r in records}) != 1 or len({r["limit_rows"] for r in records}) != 1:
        raise SystemExit("[FAILED] shards were translated with different model revisions or row limits")
    frame = pd.concat(frames).sort_values("item_index").reset_index(drop=True)
    expected = all_items(records[0]["limit_rows"])
    if len(frame) != len(expected) or not (frame["item_index"].to_numpy() == expected["item_index"].to_numpy()).all():
        raise SystemExit("[FAILED] the shards do not cover every item exactly once")
    for col in ("corpus", "split", "row", "labels", "source_text"):
        if not (frame[col].astype(str).to_numpy() == expected[col].astype(str).to_numpy()).all():
            raise SystemExit(f"[FAILED] column {col} differs from the prepared corpora")
    frame["forward_clean"] = [clean_output(t, lang) for t, lang in zip(frame["forward_text"], frame["target_lang"])]
    frame["backward_clean"] = [clean_output(t, lang) for t, lang in zip(frame["backward_text"], frame["source_lang"])]

    report = {"mt_model": records[0]["mt_model"], "mt_revision": records[0]["mt_revision"],
              "labse_model": records[0]["labse_model"], "labse_revision": records[0]["labse_revision"],
              "settings": records[0]["settings"], "limit_rows": records[0]["limit_rows"],
              "shards": [{k: r[k] for k in ("shard", "items", "sha256", "wall_seconds", "host")} for r in records],
              "tasks": {}}
    for task, (corpus, step) in TASK_SOURCES.items():
        out_dir = DATA_DIR / task
        out_dir.mkdir(parents=True, exist_ok=True)
        report["tasks"][task] = {}
        for split in SPLITS:
            d = frame[(frame["corpus"] == corpus) & (frame["split"] == split)]
            out = pd.DataFrame({"text": d[f"{step}_text"].to_numpy(), "labels": d["labels"].to_numpy(),
                                "text_clean": d[f"{step}_clean"].to_numpy(), "source_row": d["row"].to_numpy()})
            kept = out[out["text_clean"].str.len() > 0]
            path = out_dir / f"{split}.csv"
            kept.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
            report["tasks"][task][split] = {"rows": len(kept), "dropped_empty": len(out) - len(kept),
                                            "offensive_share": round(float(kept["labels"].mean()), 4),
                                            "sha256": sha256_file(path)}
    report["created"] = datetime.now(timezone.utc).isoformat()
    PREPARATION_RECORD.write_text(json.dumps(report, indent=2), encoding="utf-8")
    quality = pd.DataFrame(quality_rows(frame))
    QUALITY_TABLE.parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(QUALITY_TABLE, index=False, float_format="%.6g", lineterminator="\n")
    for task, splits in report["tasks"].items():
        print(task, {s: (v["rows"], v["dropped_empty"], v["sha256"][:12]) for s, v in splits.items()}, flush=True)
    summary = quality[(quality["split"] == "all") & (quality["label"].astype(str) == "all")]
    for row in summary.itertuples():
        print(f"{row.corpus}: LaBSE forward {row.labse_forward_mean:.3f} (p05 {row.labse_forward_p05:.3f}), "
              f"round-trip chrF++ {row.roundtrip_chrf_pp:.1f}, target script {row.forward_target_script_share:.3f}, "
              f"degenerate {row.forward_degenerate_rate:.4f}", flush=True)
    print(f"Wrote {PREPARATION_RECORD} and {QUALITY_TABLE}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Machine translation of the offensive source corpora.")
    sub = parser.add_subparsers(dest="step", required=True)
    for name in ("translate", "assemble"):
        p = sub.add_parser(name)
        p.add_argument("--num-shards", type=int, required=True)
        p.add_argument("--store", default="hub", help="hub or local:<dir>")
        p.add_argument("--wait-hours", type=float, default=6.0)
    t = sub.choices["translate"]
    t.add_argument("--shard", type=int, required=True)
    t.add_argument("--wait", action="store_true", help="Return only when all shards are in the store.")
    t.add_argument("--devices", default=None, help="Comma-separated CUDA device ids, one worker each.")
    t.add_argument("--device", default=None, help="Force a device (cpu for tests).")
    t.add_argument("--limit-rows", type=int, default=None, help="First rows of every split only (tests).")
    for p in (t, sub.add_parser("worker")):
        p.add_argument("--mt-model", default=MT_MODEL)
        p.add_argument("--labse-model", default=LABSE_MODEL)
    w = sub.choices["worker"]
    w.add_argument("--input", required=True)
    w.add_argument("--output", required=True)
    w.add_argument("--device", default=None)
    w.add_argument("--load-after", default=None, help="Wait for this file before loading the model.")
    w.add_argument("--loaded-flag", default=None, help="Write this file once the model is loaded.")
    args = parser.parse_args()
    if args.step == "worker":
        run_worker(args)
    elif args.step == "translate":
        if not 0 <= args.shard < args.num_shards:
            raise SystemExit("--shard must lie in 0..num_shards-1")
        run_translate(args)
    else:
        run_assemble(args)


if __name__ == "__main__":
    main()
