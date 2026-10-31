"""Contamination checks for the open LLMs: were the evaluation texts part of their pretraining data?

Text sets (rows of the processed split files; samples are fixed by seed):
  offmix3l, sentmix3l      the code-mixed test sets whose LLM results are under suspicion (public since 2023)
  bdshs, sentnob           public in-domain test sets (2021-2022), likely seen by every model
  sylheti                  the authors' Sylheti translations, never published: certain non-members
  chaos_banglish, mixtox   the natural code-mixed targets
1. Likelihood statistics, raw text with the model's BOS token, no prompt (`{set}_likelihood.npz`):
   mean token log-probability, Min-K% Prob (Shi et al., 2024; mean of the lowest MIN_K share of token log-probs),
   Min-K%++ (Zhang et al., 2024; token log-prob standardised by the mean and sd of the log-prob under the model's own
   next-token distribution, lowest MIN_K share), and the zlib ratio (Carlini et al., 2021); each also for a
   word-shuffled copy of the text (same words and tokens, no memorisable order) as a paired reference.
2. Guided versus general instance completion (Golchin & Surdeanu, 2024; `{set}_completion.jsonl`): the first half of
   the words is given with (guided) or without (general) the dataset name and split, greedy decoding, and the
   continuation is compared with the true second half (word-level ROUGE-L F1, exact five-word prefix). Memorised
   instances are completed better under the guided prompt.
Outputs: run-outputs/llm_membership_inference/{model}/ with a marker per set and part (resumable).

    python -m src.llm_membership_inference --models qwen3_8b llama3_1_8b_instruct --deadline-hours 10
"""

import argparse
import json
import os
import random
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BASE_DIR, DATA_DIR, HUB_OUTPUT_REPO
from src.llm_inference import LLM_MODELS, describe_error, load_model, model_revision, render, stable_seed

PREFIX = "llm_membership_inference"
WORK_DIR = BASE_DIR / "results" / PREFIX
MIN_K = 0.2
LIKELIHOOD_ITEMS = 1000
COMPLETION_ITEMS = 250
COMPLETION_MIN_WORDS = 8
COMPLETION_REFERENCE_WORDS = 30
SCORE_TOKEN_BUDGET = 1024
MAX_TEXT_TOKENS = 512
GENERATION_BATCH = 16
TEXT_SETS = {
    "offmix3l": ("offensive", ["d2_codemixed_cal.csv", "d2_codemixed_test.csv"], "OffMix-3L", "test"),
    "sentmix3l": ("sentiment", ["d2_codemixed_cal.csv", "d2_codemixed_test.csv"], "SentMix-3L", "test"),
    "bdshs": ("offensive", ["test.csv"], "BD-SHS", "test"),
    "sentnob": ("sentiment", ["test.csv"], "SentNoB", "test"),
    "sylheti": ("sentiment", ["d1_dialectal_cal.csv", "d1_dialectal_test.csv"], "SentNoB-Sylheti", "test"),
    "chaos_banglish": ("offensive", ["d4_banglish_cal.csv", "d4_banglish_test.csv"], "Code-mixed Chaos", "train"),
    "mixtox": ("offensive", ["d5_mixtox_cal.csv", "d5_mixtox_test.csv"], "MixTox-SA", "train"),
}
GUIDED = ("You are provided with the first piece of an instance from the {split} split of the {dataset} dataset. "
          "Finish the second piece of the instance as exactly appeared in the dataset. Only rely on the original "
          "form of the instance in the dataset to finish the second piece.\n\nFirst piece: {first}\n\nSecond piece:")
GENERAL = ("Finish the second piece based on the first piece, such that these two pieces become a single "
           "instance.\n\nFirst piece: {first}\n\nSecond piece:")


def load_set(name: str) -> pd.DataFrame:
    task, files, _, _ = TEXT_SETS[name]
    frames = []
    for f in files:
        df = pd.read_csv(DATA_DIR / task / f, usecols=["text_clean", "labels"])
        frames.append(pd.DataFrame({"file": f, "row": np.arange(len(df)), "text": df.text_clean.astype(str)}))
    return pd.concat(frames, ignore_index=True)


def sample(frame: pd.DataFrame, n: int, key: str) -> pd.DataFrame:
    if len(frame) <= n:
        return frame
    rng = np.random.default_rng(stable_seed(f"membership:{key}"))
    return frame.iloc[np.sort(rng.choice(len(frame), n, replace=False))]


def shuffled(text: str, key: str) -> str:
    words = text.split()
    random.Random(stable_seed(f"shuffle:{key}")).shuffle(words)
    return " ".join(words)


def lcs(a: list, b: list) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(candidate: str, reference: str) -> float:
    c, r = candidate.lower().split(), reference.lower().split()
    if not c or not r:
        return 0.0
    m = lcs(c, r)
    return 0.0 if m == 0 else 2 * m / (len(c) + len(r))


def token_statistics(net, tok, texts: list, log) -> dict:
    """Per text: mean log-prob, Min-K%, Min-K%++ and token count (BOS prepended when the tokenizer has one)."""
    import torch

    device = net.get_input_embeddings().weight.device
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    encoded = [(bos + tok(t, add_special_tokens=False)["input_ids"])[:MAX_TEXT_TOKENS] for t in texts]
    out = {k: np.full(len(texts), np.nan) for k in ("mean_logprob", "min_k", "min_k_plus_plus", "n_tokens")}
    order = np.argsort([-len(e) for e in encoded], kind="stable")
    i = 0
    while i < len(order):
        length = len(encoded[order[i]])
        b = max(1, min(64, SCORE_TOKEN_BUDGET // max(length, 1), len(order) - i))
        rows = order[i:i + b]
        ids = torch.full((b, length), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((b, length), dtype=torch.long)
        for r, j in enumerate(rows):
            ids[r, :len(encoded[j])] = torch.tensor(encoded[j])
            mask[r, :len(encoded[j])] = 1
        with torch.inference_mode():
            logits = net(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False).logits
            logits = logits[:, :-1].float()
            lsm = torch.log_softmax(logits, dim=-1)
            target = ids[:, 1:].to(lsm.device)
            token_lp = lsm.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            probs = lsm.exp()
            mu = (probs * lsm).sum(-1)
            sigma = ((probs * lsm.pow(2)).sum(-1) - mu.pow(2)).clamp(min=1e-8).sqrt()
            z = ((token_lp - mu) / sigma).cpu().numpy()
            token_lp = token_lp.cpu().numpy()
        del logits, lsm, probs
        for r, j in enumerate(rows):
            n = len(encoded[j]) - 1
            if n < 1:
                continue
            lp, zz = token_lp[r, :n], z[r, :n]
            k = max(1, int(np.ceil(MIN_K * n)))
            out["mean_logprob"][j] = float(lp.mean())
            out["min_k"][j] = float(np.sort(lp)[:k].mean())
            out["min_k_plus_plus"][j] = float(np.sort(zz)[:k].mean())
            out["n_tokens"][j] = n
        i += b
    return out


def likelihood_part(net, tok, name: str, out: Path, log) -> dict:
    frame = sample(load_set(name), LIKELIHOOD_ITEMS, name)
    texts = frame.text.tolist()
    keys = [f"{name}:{f}:{r}" for f, r in zip(frame.file, frame.row)]
    t0 = time.time()
    original = token_statistics(net, tok, texts, log)
    perm = token_statistics(net, tok, [shuffled(t, k) for t, k in zip(texts, keys)], log)
    zlib_bits = np.array([len(zlib.compress(t.encode("utf-8"))) for t in texts], dtype=float)
    arrays = {"source_file": frame.file.to_numpy().astype(str), "row": frame.row.to_numpy(), "zlib_bytes": zlib_bits}
    arrays.update({k: v for k, v in original.items()})
    arrays.update({f"shuffled_{k}": v for k, v in perm.items()})
    arrays["zlib_ratio"] = -original["mean_logprob"] * original["n_tokens"] / zlib_bits
    path = out / f"{name}_likelihood.npz"
    np.savez(path, **arrays)
    log(f"likelihood {name}: {len(texts)} texts in {(time.time() - t0) / 60:.1f} min; mean Min-K%++ "
        f"{np.nanmean(original['min_k_plus_plus']):.3f} (shuffled {np.nanmean(perm['min_k_plus_plus']):.3f})")
    return {f"{name}_likelihood.npz": path}


def generate(net, tok, prompts: list, max_new: list, log) -> list:
    import torch

    device = net.get_input_embeddings().weight.device
    tok.padding_side = "left"
    order = np.argsort([-len(p) for p in prompts], kind="stable")
    outputs = [""] * len(prompts)
    for s in range(0, len(order), GENERATION_BATCH):
        rows = order[s:s + GENERATION_BATCH]
        batch = tok([prompts[j] for j in rows], return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.inference_mode():
            gen = net.generate(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device),
                               max_new_tokens=int(max(max_new[j] for j in rows)), do_sample=False,
                               pad_token_id=tok.pad_token_id)
        new = gen[:, batch["input_ids"].shape[1]:]
        for r, j in enumerate(rows):
            outputs[j] = tok.decode(new[r], skip_special_tokens=True).strip()
    return outputs


def completion_part(net, tok, model: str, name: str, out: Path, log) -> dict:
    _, _, dataset, split = TEXT_SETS[name]
    frame = load_set(name)
    frame = frame[frame.text.str.split().str.len() >= COMPLETION_MIN_WORDS]
    frame = sample(frame, COMPLETION_ITEMS, f"{name}:completion")
    kwargs = LLM_MODELS[model].get("template_kwargs", {})
    items, prompts, max_new = [], [], []
    for f, r, text in zip(frame.file, frame.row, frame.text):
        words = text.split()
        half = len(words) // 2
        first, second = " ".join(words[:half]), " ".join(words[half:half + COMPLETION_REFERENCE_WORDS])
        budget = min(160, 16 + 4 * len(second.split()))
        items.append({"file": f, "row": int(r), "first": first, "reference": second})
        for kind, template in (("guided", GUIDED), ("general", GENERAL)):
            prompts.append(render(tok, template.format(split=split, dataset=dataset, first=first), kwargs))
            max_new.append(budget)
    t0 = time.time()
    texts = generate(net, tok, prompts, max_new, log)
    for i, item in enumerate(items):
        for k, kind in enumerate(("guided", "general")):
            cont = texts[2 * i + k]
            item[f"{kind}_completion"] = cont
            item[f"{kind}_rouge_l"] = rouge_l(" ".join(cont.split()[:len(item["reference"].split())]), item["reference"])
            item[f"{kind}_exact_prefix5"] = cont.lower().split()[:5] == item["reference"].lower().split()[:5]
    path = out / f"{name}_completion.jsonl"
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n", encoding="utf-8")
    g = np.mean([x["guided_rouge_l"] for x in items])
    h = np.mean([x["general_rouge_l"] for x in items])
    log(f"completion {name}: {len(items)} items in {(time.time() - t0) / 60:.1f} min; ROUGE-L guided {g:.3f}, "
        f"general {h:.3f}; exact 5-word prefix guided {np.mean([x['guided_exact_prefix5'] for x in items]):.3f}")
    return {f"{name}_completion.jsonl": path}


class HubStore:
    def __init__(self):
        from huggingface_hub import HfApi

        self.api = HfApi(token=os.environ.get("HF_TOKEN"))

    def done(self, model: str, marker: str) -> bool:
        try:
            return self.api.file_exists(HUB_OUTPUT_REPO, f"{PREFIX}/{model}/{marker}", repo_type="dataset")
        except Exception:
            return False

    def commit(self, model: str, files: dict) -> bool:
        from huggingface_hub import CommitOperationAdd

        ops = [CommitOperationAdd(path_in_repo=f"{PREFIX}/{model}/{n}", path_or_fileobj=str(p)) for n, p in files.items()]
        for attempt in range(6):
            try:
                self.api.create_commit(HUB_OUTPUT_REPO, ops, repo_type="dataset",
                                       commit_message=f"Add membership-inference outputs for {model}")
                return True
            except Exception as e:
                print(f"  [UPLOAD RETRY {attempt + 1}/6] {model}: {describe_error(e)}", flush=True)
                time.sleep(min(300, 15 * 2 ** attempt))
        return False


class LocalStore:
    def __init__(self, root: str):
        self.root = Path(root)

    def done(self, model: str, marker: str) -> bool:
        return (self.root / PREFIX / model / marker).exists()

    def commit(self, model: str, files: dict) -> bool:
        target = self.root / PREFIX / model
        target.mkdir(parents=True, exist_ok=True)
        for n, p in files.items():
            (target / n).write_bytes(Path(p).read_bytes())
        return True


def main():
    import torch

    parser = argparse.ArgumentParser(description="Membership-inference and completion tests for the LLMs.")
    parser.add_argument("--models", nargs="+", default=["qwen3_8b"], choices=sorted(LLM_MODELS))
    parser.add_argument("--sets", nargs="+", default=list(TEXT_SETS), choices=list(TEXT_SETS))
    parser.add_argument("--parts", nargs="+", default=["likelihood", "completion"], choices=["likelihood", "completion"])
    parser.add_argument("--deadline-hours", type=float, default=11.0)
    parser.add_argument("--store", default="hub", help="'hub' or 'local:<dir>' (dry runs)")
    parser.add_argument("--model-path", action="append", default=[], help="model_key=local_dir (dry runs)")
    args = parser.parse_args()
    store = HubStore() if args.store == "hub" else LocalStore(args.store.split(":", 1)[1])
    local = dict(p.split("=", 1) for p in args.model_path)
    deadline = time.time() + args.deadline_hours * 3600
    log = lambda msg: print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)
    n_gpus = torch.cuda.device_count()
    log(f"device {'cuda x' + str(n_gpus) if n_gpus else 'cpu'}; models {args.models}; sets {args.sets}")
    for model in args.models:
        todo = [(name, part) for part in args.parts for name in args.sets
                if not store.done(model, f"{name}_{part}_complete.json")]
        if not todo:
            log(f"skip {model} (complete)")
            continue
        if time.time() > deadline - 1800:
            log("deadline reached; stopping")
            break
        source = local.get(model, LLM_MODELS[model]["hf_id"])
        t0 = time.time()
        net, tok = load_model(model, source, min(n_gpus, LLM_MODELS[model]["gpus"]))
        revision = model_revision(model)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        log(f"loaded {model} in {time.time() - t0:.0f} s")
        out = WORK_DIR / model
        out.mkdir(parents=True, exist_ok=True)
        for name, part in todo:
            if time.time() > deadline - 900:
                log("deadline reached; stopping")
                break
            files = (likelihood_part(net, tok, name, out, log) if part == "likelihood"
                     else completion_part(net, tok, model, name, out, log))
            marker = out / f"{name}_{part}_complete.json"
            marker.write_text(json.dumps({"model": model, "hf_id": LLM_MODELS[model]["hf_id"], "revision": revision,
                                          "set": name, "part": part, "min_k": MIN_K,
                                          "completed": datetime.now(timezone.utc).isoformat()}))
            files[marker.name] = marker
            log(f"upload {model} {name} {part}: {'ok' if store.commit(model, files) else 'FAILED'}")
        del net
        if n_gpus:
            torch.cuda.empty_cache()
    log("finished")


if __name__ == "__main__":
    main()
