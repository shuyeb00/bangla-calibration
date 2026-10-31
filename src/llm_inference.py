"""Zero- and few-shot classification with open instruction-tuned LLMs on every evaluation split.

The class scores are the log-probabilities of each label word's token sequence at the answer position
(one forward pass when every label is a single token) (the same prompt for every split and condition; English instructions,
Bangla or code-mixed text). Few-shot prompts hold `per_class` labelled demonstrations per class drawn from
the source training split with a fixed seed. Splits larger than `cap` items are scored on a fixed random
subset first (noise splits reuse the test subset and Bangla-script splits the subset of their romanised
counterpart, since their rows align); the rest of
the split is a separate, lower-priority job (`{split}_remainder`).

Outputs per run (`llm_runs/{task}_{model}_{setting}/` in the run-outputs Hub repo, and locally):
  {split}_logits.npy      label log-probabilities, float32 (n, n_classes); softmax gives class probabilities
  {split}_labels.npy      gold labels
  {split}_indices.npy     row indices into the split file
  {split}_label_mass.npy  probability of the label tokens under the full vocabulary
  {split}_complete.json   written last (counts, tokens, throughput)
  config.json             model, revision, dtype, prompt, label tokens, demonstrations, versions

Jobs run in priority order (core zero- and few-shot, then remainders, then extra demonstration seeds);
models that fit one GPU run one worker per GPU, larger models one worker across all GPUs. Finished jobs
are skipped through the Hub markers, so a session can be rerun to continue. No job is started that is not
expected to finish before the deadline.

Target demonstrations (--target-demonstrations; in-context learning with labelled target items): for every target
test split, PER_CLASS demonstrations per class are drawn from the TARGET_DEMO_LABELS labelled target calibration items
of resample s of src/shift_recalibration.subset_indices (calibration pool = the LLM subset of the calibration split),
i.e. from the same labels the few-label methods use at n = TARGET_DEMO_LABELS. Setting few_shot_target_seed{s}, output
prefix llm_target_demonstrations/, scored on TARGET_DEMO_CAP items of the primary test subset where that subset was
drawn from a larger split (all items otherwise).

Prompt variants (--prompt-variants, zero shot only): the same answer format with another instruction, run as
setting zero_shot_{variant} under the separate prefix llm_prompt_variants/ so that the analyses of llm_runs/ are
unchanged. They test whether the definition of offensiveness moves the decision threshold (OLID-style or broad
definition) and how sensitive the scores are to the wording of the instruction (paraphrase).

    python -m src.llm_inference --models qwen3_8b qwen3_5_9b --deadline-hours 11
    python -m src.llm_inference --models qwen3_8b --tasks offensive --prompt-variants olid_definition \
        broad_definition paraphrase --splits cal test shift_d2_test shift_d2_cal --no-remainders
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BASE_DIR, DATA_DIR, HUB_OUTPUT_REPO, TASKS, get_natural_codemixed_paths, \
    get_shift_conditions_paths

LLM_MODELS = {
    "qwen3_8b": {"hf_id": "Qwen/Qwen3-8B", "dtype": "float16", "gpus": 2,
                 "template_kwargs": {"enable_thinking": False}, "tokens_per_second": 1500},
    "qwen3_4b_instruct": {"hf_id": "Qwen/Qwen3-4B-Instruct-2507", "dtype": "float16", "gpus": 1,
                          "tokens_per_second": 2500},
    "llama3_1_8b_instruct": {"hf_id": "meta-llama/Llama-3.1-8B-Instruct", "dtype": "float16", "gpus": 2,
                             "tokens_per_second": 1500},
    # Gemma-3 architecture: fp16 activations can overflow on T4, so the 1B model runs in fp32.
    "tigerllm_1b_instruct": {"hf_id": "md-nishat-008/TigerLLM-1B-it", "dtype": "float32", "gpus": 1,
                             "tokens_per_second": 3000},
    "qwen3_5_9b": {"hf_id": "Qwen/Qwen3.5-9B", "dtype": "float16", "gpus": 2, "model_class": "image_text_to_text",
                   "template_kwargs": {"enable_thinking": False}, "tokens_per_second": 1200},
}

TASK_PROMPTS = {
    "sentiment": {
        "labels": {0: "Neutral", 1: "Positive", 2: "Negative"},
        "instruction": ("You are given a text in Bangla. It may contain regional dialect, informal spelling, or "
                        "words in English or Hindi. Classify the sentiment that the text expresses."),
        "answer": "Answer with exactly one word: Positive, Negative, or Neutral.",
    },
    "offensive": {
        "labels": {0: "No", 1: "Yes"},
        "instruction": ("You are given a text in Bangla. It may contain regional dialect, informal spelling, or "
                        "words in English or Hindi. Decide whether the text is offensive, that is, whether it "
                        "contains hate speech, insults, abuse, threats, or vulgar language."),
        "answer": "Answer with exactly one word: Yes if the text is offensive, No if it is not.",
    },
}

# Alternative instructions for the offensive task (answer format unchanged). olid_definition follows the
# annotation guideline of OLID (Zampieri et al., 2019), from which OffMix-3L's training data derive;
# broad_definition also counts hostile or harshly critical remarks without abusive words; paraphrase restates
# the default BD-SHS-style definition in other words.
PROMPT_VARIANTS = {
    "offensive": {
        "olid_definition": ("You are given a text in Bangla. It may contain regional dialect, informal spelling, or "
                            "words in English or Hindi. Decide whether the text is offensive, that is, whether it "
                            "contains any form of non-acceptable language (profanity) or a targeted offense, "
                            "which can be veiled or direct, including insults, threats, and posts containing "
                            "profane language or swear words."),
        "broad_definition": ("You are given a text in Bangla. It may contain regional dialect, informal spelling, or "
                             "words in English or Hindi. Decide whether the text is offensive, that is, whether it "
                             "could hurt, upset or anger a person or group, including hostile, mocking or harshly "
                             "critical remarks even when they contain no abusive words."),
        "paraphrase": ("The following text is written in Bangla and may include dialect, non-standard spelling, or "
                       "English and Hindi words. Your task is to judge whether it is offensive: does it insult, "
                       "abuse or threaten anyone, express hate, or use vulgar words?"),
    },
}
OUTPUT_PREFIXES = ("llm_runs", "llm_prompt_variants", "llm_target_demonstrations")

SPLIT_PRIORITY = ["cal", "test", "shift_d2_test", "shift_d2_cal", "shift_d1_test", "shift_d1_cal",
                  "shift_d1_barishal_test", "shift_d1_chittagong_test", "shift_d1_noakhali_test",
                  "shift_d1_barishal_cal", "shift_d1_chittagong_cal", "shift_d1_noakhali_cal",
                  "shift_d3_noise10", "shift_d3_noise05", "shift_d3_noise20",
                  "shift_d4_banglish_test", "shift_d4_banglish_cal", "shift_d4_banglish_bangla_script_test",
                  "shift_d4_banglish_bangla_script_cal", "shift_d5_mixtox_test", "shift_d5_mixtox_cal"]
CAP = 1500
TARGET_DEMO_LABELS = 50
TARGET_DEMO_MAX_CHARS = 600
TARGET_DEMO_CAP = 600
PER_CLASS = 3
DEMO_CHARS = (20, 150)
TOKEN_BUDGET = 12288
MAX_BATCH = 64
HUB_PREFIX = "llm_runs"
WORK_DIR = BASE_DIR / "results" / "llm_runs"


def set_output_prefix(prefix: str) -> None:
    global HUB_PREFIX, WORK_DIR
    if prefix not in OUTPUT_PREFIXES:
        raise ValueError(f"output prefix must be one of {OUTPUT_PREFIXES}")
    HUB_PREFIX, WORK_DIR = prefix, BASE_DIR / "results" / prefix


def instruction(task: str, setting: str) -> str:
    for variant, text in PROMPT_VARIANTS.get(task, {}).items():
        if setting == f"zero_shot_{variant}":
            return text
    return TASK_PROMPTS[task]["instruction"]


def split_files(task: str) -> dict:
    files = {"cal": DATA_DIR / task / "cal.csv", "test": DATA_DIR / task / "test.csv"}
    files.update(get_shift_conditions_paths(task))
    files.update(get_natural_codemixed_paths(task))
    return {k: files[k] for k in SPLIT_PRIORITY if k in files}


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def subset_indices(task: str, split: str, n: int, cap: int) -> tuple:
    if n <= cap:
        return np.arange(n), np.array([], dtype=int)
    base = "test" if split.startswith("shift_d3_noise") else split.replace("_bangla_script", "")
    primary = np.sort(np.random.default_rng(stable_seed(f"{task}:{base}")).choice(n, cap, replace=False))
    return primary, np.setdiff1d(np.arange(n), primary)


def demonstrations(task: str, seed: int, per_class: int) -> list:
    train = pd.read_csv(DATA_DIR / task / "train.csv")
    text = train.text_clean.astype(str)
    ok = text.str.len().between(*DEMO_CHARS) & ~text.str.contains("�")
    rng = np.random.default_rng(stable_seed(f"{task}:demonstrations:{seed}"))
    picks = []
    for label in sorted(TASK_PROMPTS[task]["labels"]):
        pool = np.flatnonzero(ok.to_numpy() & (train.labels.to_numpy() == label))
        picks += [int(i) for i in rng.choice(pool, per_class, replace=False)]
    picks = [picks[i] for i in rng.permutation(len(picks))]
    return [{"train_row": i, "text": text.iloc[i], "label": int(train.labels.iloc[i])} for i in picks]


def target_cal_split(test_split: str) -> str:
    return test_split[: -len("_test")] + "_cal"


def target_demonstrations(task: str, test_split: str, seed: int, per_class: int) -> list:
    """Demonstrations from the labelled target calibration items of resample `seed` at n = TARGET_DEMO_LABELS: per
    class, random items of at most TARGET_DEMO_MAX_CHARS characters (the shortest ones if too few are that short);
    a class with fewer labelled items contributes all of them."""
    from src.shift_recalibration import subset_indices as label_subset

    cal_split = target_cal_split(test_split)
    cal = pd.read_csv(split_files(task)[cal_split])
    pool, _ = subset_indices(task, cal_split, len(cal), CAP)
    labelled = pool[label_subset(len(pool), min(TARGET_DEMO_LABELS, len(pool)), seed)]
    text, y = cal.text_clean.astype(str).to_numpy(), cal.labels.to_numpy()
    base = test_split.replace("_bangla_script", "")
    rng = np.random.default_rng(stable_seed(f"{task}:{base}:target_demonstrations:{seed}"))
    picks = []
    for label in sorted(TASK_PROMPTS[task]["labels"]):
        rows = labelled[y[labelled] == label]
        short = rows[np.char.str_len(text[rows].astype(str)) <= TARGET_DEMO_MAX_CHARS]
        if len(short) >= per_class:
            chosen = rng.choice(short, per_class, replace=False)
        else:
            chosen = rows[np.argsort(np.char.str_len(text[rows].astype(str)), kind="stable")][:per_class]
        picks += [int(i) for i in chosen]
    picks = [picks[i] for i in rng.permutation(len(picks))]
    return [{"cal_split": cal_split, "cal_row": i, "text": text[i], "label": int(y[i])} for i in picks]


def target_eval_rows(task: str, split: str, n: int) -> np.ndarray:
    primary, _ = subset_indices(task, split, n, CAP)
    if len(primary) < CAP or len(primary) <= TARGET_DEMO_CAP:
        return primary
    base = split.replace("_bangla_script", "")
    rng = np.random.default_rng(stable_seed(f"{task}:{base}:target_demonstration_items"))
    return np.sort(rng.choice(primary, TARGET_DEMO_CAP, replace=False))


def user_message(task: str, text: str, demos: list, setting: str = "zero_shot") -> str:
    spec = TASK_PROMPTS[task]
    head = instruction(task, setting)
    if not demos:
        return f"{head}\n\nText: {text}\n\n{spec['answer']}"
    shots = "\n\n".join(f"Text: {d['text']}\nAnswer: {spec['labels'][d['label']]}" for d in demos)
    return (f"{head}\n\nExamples:\n\n{shots}\n\nNow classify this text.\n\n"
            f"Text: {text}\n\n{spec['answer']}")


def render(tokenizer, content: str, template_kwargs: dict) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=True, **template_kwargs)
    return f"{content}\nAnswer:"


def label_sequences(tokenizer, prompt: str, labels: dict, chat: bool) -> list:
    """Token sequence of every label word as it continues the prompt (in class order)."""
    prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    seqs = []
    for label in [labels[k] for k in sorted(labels)]:
        full = tokenizer(prompt + (label if chat else " " + label), add_special_tokens=False)["input_ids"]
        if full[:len(prefix)] != prefix or len(full) <= len(prefix):
            raise ValueError(f"label {label!r} does not start a new token after the prompt")
        seqs.append(full[len(prefix):])
    if len({tuple(q) for q in seqs}) != len(seqs):
        raise ValueError(f"label words have identical token sequences: {seqs}")
    return seqs


def build_jobs(models: list, tasks: list, seeds: list, cap: int, few_shot_remainders: bool = True,
               splits: list = None, remainders: bool = True) -> list:
    """Priority order: per model, core (zero shot, first few-shot seed) on the primary subsets; then for every
    model the remainders of zero shot; then extra seeds (one seed complete before the next) and few-shot
    remainders. `splits` restricts the jobs to the named splits; remainders=False drops every remainder job."""
    tiers = []
    for model in models:
        core, rest, extra = [], [], []
        for task in tasks:
            for split, path in split_files(task).items():
                if splits is not None and split not in splits:
                    continue
                n = len(pd.read_csv(path, usecols=["labels"]))
                _, remainder = subset_indices(task, split, n, cap)
                for s_i, setting in enumerate(["zero_shot"] + [f"few_shot_seed{s}" for s in seeds]):
                    job = {"model": model, "task": task, "setting": setting, "split": split, "part": "primary"}
                    (core if s_i <= 1 else extra).append(job)
                    if remainders and len(remainder) and (s_i == 0 or few_shot_remainders):
                        tier = rest if s_i == 0 else extra
                        tier.append({**job, "part": "remainder"})
        settings = ["zero_shot"] + [f"few_shot_seed{s}" for s in seeds]
        extra.sort(key=lambda j: (j["part"] == "remainder", settings.index(j["setting"])))
        tiers.append((core, rest, extra))
    order = [j for t in tiers for j in t[0]] + [j for t in tiers for j in t[1]] + [j for t in tiers for j in t[2]]
    for j in order:
        j["run_id"] = f"{j['task']}_{j['model']}_{j['setting']}"
        j["name"] = j["split"] if j["part"] == "primary" else f"{j['split']}_remainder"
        j["job_id"] = f"{j['run_id']}__{j['name']}"
    return order


def build_variant_jobs(models: list, tasks: list, variants: list, cap: int, splits: list = None) -> list:
    """Zero-shot prompt-variant jobs on the primary item subsets, model by model and variant by variant."""
    jobs = []
    for model in models:
        for task in tasks:
            unknown = set(variants) - set(PROMPT_VARIANTS.get(task, {}))
            if unknown:
                raise ValueError(f"no prompt variants {sorted(unknown)} for task {task}")
            for variant in variants:
                for split in split_files(task):
                    if splits is None or split in splits:
                        jobs.append({"model": model, "task": task, "setting": f"zero_shot_{variant}", "split": split,
                                     "part": "primary"})
    for j in jobs:
        j["run_id"] = f"{j['task']}_{j['model']}_{j['setting']}"
        j["name"] = j["split"]
        j["job_id"] = f"{j['run_id']}__{j['name']}"
    return jobs


def build_target_demo_jobs(models: list, tasks: list, seeds: list, splits: list = None) -> list:
    """Per model, one seed complete over every target test split before the next seed."""
    jobs = []
    for model in models:
        for seed in seeds:
            for task in tasks:
                files = split_files(task)
                for split in files:
                    if not split.endswith("_test") or target_cal_split(split) not in files:
                        continue
                    if splits is None or split in splits:
                        jobs.append({"model": model, "task": task, "setting": f"few_shot_target_seed{seed}",
                                     "split": split, "part": "target_demonstration"})
    for j in jobs:
        j["run_id"] = f"{j['task']}_{j['model']}_{j['setting']}"
        j["name"] = j["split"]
        j["job_id"] = f"{j['run_id']}__{j['name']}"
    return jobs


def planned_jobs(args) -> list:
    if args.target_demonstrations:
        return build_target_demo_jobs(args.models, args.tasks, args.few_shot_seeds, args.splits)
    if args.prompt_variants:
        return build_variant_jobs(args.models, args.tasks, args.prompt_variants, args.cap, args.splits)
    return build_jobs(args.models, args.tasks, args.few_shot_seeds, args.cap, not args.skip_few_shot_remainders,
                      args.splits, not args.no_remainders)


class HubStore:
    def __init__(self, repo: str):
        from huggingface_hub import HfApi

        self.api, self.repo = HfApi(token=os.environ.get("HF_TOKEN")), repo

    def done(self, job: dict) -> bool:
        try:
            return self.api.file_exists(self.repo, f"{HUB_PREFIX}/{job['run_id']}/{job['name']}_complete.json",
                                        repo_type="dataset")
        except Exception as e:
            print(f"  marker check failed ({type(e).__name__}); treating {job['job_id']} as not done", flush=True)
            return False

    def stored_indices(self, job: dict):
        from huggingface_hub import hf_hub_download

        try:
            return np.load(hf_hub_download(self.repo, f"{HUB_PREFIX}/{job['run_id']}/{job['name']}_indices.npy",
                                           repo_type="dataset", token=os.environ.get("HF_TOKEN")))
        except Exception:
            return None

    def commit(self, job: dict, files: dict) -> bool:
        from huggingface_hub import CommitOperationAdd

        ops = [CommitOperationAdd(path_in_repo=f"{HUB_PREFIX}/{job['run_id']}/{name}", path_or_fileobj=str(path))
               for name, path in files.items()]
        for attempt in range(6):
            try:
                self.api.create_commit(self.repo, ops, repo_type="dataset",
                                       commit_message=f"Add LLM outputs {job['job_id']}")
                return True
            except Exception as e:
                print(f"  [UPLOAD RETRY {attempt + 1}/6] {job['job_id']}: {describe_error(e)}", flush=True)
                time.sleep(min(300, 15 * 2 ** attempt))
        return False


class LocalStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def done(self, job: dict) -> bool:
        return (self.root / HUB_PREFIX / job["run_id"] / f"{job['name']}_complete.json").exists()

    def stored_indices(self, job: dict):
        path = self.root / HUB_PREFIX / job["run_id"] / f"{job['name']}_indices.npy"
        return np.load(path) if path.exists() else None

    def commit(self, job: dict, files: dict) -> bool:
        target = self.root / HUB_PREFIX / job["run_id"]
        target.mkdir(parents=True, exist_ok=True)
        for name, path in files.items():
            (target / name).write_bytes(Path(path).read_bytes())
        return True


def describe_error(e: Exception) -> str:
    response = getattr(e, "response", None)
    server = ""
    if response is not None:
        server = f" [HTTP {getattr(response, 'status_code', '?')}] {str(getattr(response, 'text', ''))[:500]}"
    return f"{type(e).__name__}: {str(e)[:300]}{server}"


def expected_indices(job: dict, cap: int) -> np.ndarray:
    n = len(pd.read_csv(split_files(job["task"])[job["split"]], usecols=["labels"]))
    if job["part"] == "target_demonstration":
        return target_eval_rows(job["task"], job["split"], n)
    primary, remainder = subset_indices(job["task"], job["split"], n, cap)
    return primary if job["part"] == "primary" else remainder


def subset_mismatch(store, job: dict, cap: int) -> bool:
    stored = store.stored_indices(job)
    return stored is None or not np.array_equal(np.asarray(stored), expected_indices(job, cap))


def make_store(spec: str):
    return HubStore(HUB_OUTPUT_REPO) if spec == "hub" else LocalStore(Path(spec.split(":", 1)[1]))


def resolve_source(model: str, local_paths: dict) -> str:
    return local_paths.get(model, LLM_MODELS[model]["hf_id"])


def load_model(model: str, source: str, n_gpus: int):
    import torch
    import transformers

    spec = LLM_MODELS[model]
    dtype = getattr(torch, spec["dtype"]) if n_gpus else torch.float32
    kwargs = {"low_cpu_mem_usage": True, "token": os.environ.get("HF_TOKEN")}
    if n_gpus > 1:
        kwargs.update(device_map="auto", max_memory={i: "13GiB" for i in range(n_gpus)})
    elif n_gpus == 1:
        kwargs["device_map"] = {"": 0}
    cls = (transformers.AutoModelForImageTextToText if spec.get("model_class") == "image_text_to_text"
           else transformers.AutoModelForCausalLM)
    try:
        net = cls.from_pretrained(source, dtype=dtype, **kwargs)
    except TypeError:
        net = cls.from_pretrained(source, torch_dtype=dtype, **kwargs)
    tok = transformers.AutoTokenizer.from_pretrained(source, token=os.environ.get("HF_TOKEN"))
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return net.eval(), tok


def last_logits(net, input_ids, attention_mask, explicit_positions: bool, keep: int):
    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}
    if explicit_positions:
        kwargs["position_ids"] = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    try:
        out = net(**kwargs, logits_to_keep=keep)
    except TypeError:
        out = net(**kwargs)
    return out.logits[:, -keep:].to(input_ids.device).float()


def score(net, tok, encoded: list, label_seqs: list, explicit_positions: bool, log) -> tuple:
    """Log-probability of every label's token sequence after each prompt. One forward pass when all labels are
    single tokens, otherwise one pass per label (the prompt followed by all but the last label token)."""
    import torch

    device = net.get_input_embeddings().weight.device
    order = np.argsort([-len(e) for e in encoded], kind="stable")
    n, k_labels = len(encoded), len(label_seqs)
    logp = np.full((n, k_labels), np.nan, dtype=np.float64)
    single = all(len(q) == 1 for q in label_seqs)
    first = torch.tensor([q[0] for q in label_seqs], device=device)
    nonfinite = 0
    for k in ([None] if single else range(k_labels)):
        suffix = [] if k is None else label_seqs[k][:-1]
        keep = 1 if k is None else len(label_seqs[k])
        cap, i = MAX_BATCH, 0
        while i < n:
            length = len(encoded[order[i]]) + len(suffix)
            b = max(1, min(cap, TOKEN_BUDGET // max(length, 1), n - i))
            rows = order[i:i + b]
            batch = torch.full((b, length), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((b, length), dtype=torch.long)
            for r, j in enumerate(rows):
                seq = encoded[j] + suffix
                batch[r, length - len(seq):] = torch.tensor(seq)
                mask[r, length - len(seq):] = 1
            try:
                with torch.inference_mode():
                    full = last_logits(net, batch.to(device), mask.to(device), explicit_positions, keep)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if b == 1:
                    raise
                cap = max(1, b // 2)
                log(f"    out of memory at batch {b} x {length} tokens; batch cap {cap}")
                continue
            nonfinite += int((~torch.isfinite(full).all(dim=-1).all(dim=-1)).sum())
            lsm = torch.log_softmax(full, dim=-1)
            if k is None:
                logp[rows] = lsm[:, -1, first].double().cpu().numpy()
            else:
                target = torch.tensor(label_seqs[k], device=device)
                logp[rows, k] = lsm[:, torch.arange(keep, device=device), target].sum(-1).double().cpu().numpy()
            i += b
    mass = np.exp(logp).sum(axis=1)
    return logp.astype(np.float32), mass.astype(np.float32), nonfinite


def model_revision(model: str):
    try:
        from huggingface_hub import model_info

        return model_info(LLM_MODELS[model]["hf_id"], token=os.environ.get("HF_TOKEN")).sha
    except Exception:
        return None


def run_config(job: dict, model: str, source: str, revision, tok, label_seqs: list, demos: list, cap: int) -> dict:
    import torch
    import transformers

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE_DIR, capture_output=True, text=True).stdout.strip()
    spec = TASK_PROMPTS[job["task"]]
    template = render(tok, user_message(job["task"], "{text}", demos, job["setting"]),
                      LLM_MODELS[model].get("template_kwargs", {}))
    seed = int(job["setting"].rsplit("seed", 1)[1]) if job["setting"].startswith("few_shot") else None
    target = job["setting"].startswith("few_shot_target")
    return {"run_id": job["run_id"], "task": job["task"], "model_key": model, "hf_id": LLM_MODELS[model]["hf_id"],
            "source": source, "revision": revision, "ft_method": job["setting"], "objective": "prompting",
            "learning_rate": 0.0, "warmup_ratio": 0.0, "seed": seed, "dtype": LLM_MODELS[model]["dtype"],
            "labels": spec["labels"], "label_token_ids": label_seqs,
            "label_tokens": [tok.convert_ids_to_tokens(q) for q in label_seqs],
            "label_scoring": "first token" if all(len(q) == 1 for q in label_seqs) else "token sequence",
            "instruction": instruction(job["task"], job["setting"]), "output_prefix": HUB_PREFIX,
            "answer_instruction": spec["answer"], "demonstrations": [] if target else demos, "per_class": PER_CLASS,
            "demonstrations_source": "target calibration split, one set per test split ({split}_demonstrations.json)"
            if target else "source training split",
            "target_demonstration_labels": TARGET_DEMO_LABELS if target else None,
            "subset_cap": cap, "prompt_template": template, "chat_template": bool(getattr(tok, "chat_template", None)),
            "template_kwargs": LLM_MODELS[model].get("template_kwargs", {}), "torch": torch.__version__,
            "transformers": transformers.__version__, "python": platform.python_version(), "code_commit": commit}


def run_job(job: dict, net, tok, model: str, source: str, revision, demos: list, cap: int, store, out_root: Path,
            tps: float, deadline: float, log) -> tuple:
    task, split = job["task"], job["split"]
    df = pd.read_csv(split_files(task)[split])
    if job["part"] == "target_demonstration":
        rows = target_eval_rows(task, split, len(df))
    else:
        primary, remainder = subset_indices(task, split, len(df), cap)
        rows = primary if job["part"] == "primary" else remainder
    texts = df.text_clean.astype(str).to_numpy()[rows]
    labels = df.labels.to_numpy()[rows].astype(np.int64)
    kwargs = LLM_MODELS[model].get("template_kwargs", {})
    prompts = [render(tok, user_message(task, t, demos, job["setting"]), kwargs) for t in texts]
    encoded = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    n_tokens = int(sum(len(e) for e in encoded))
    remaining = deadline - time.time()
    if n_tokens / tps > 0.9 * remaining:
        log(f"stop before {job['job_id']}: needs about {n_tokens / tps / 60:.0f} min, {remaining / 60:.0f} min left")
        return "deadline", tps
    chat = bool(getattr(tok, "chat_template", None))
    label_seqs = label_sequences(tok, prompts[0], TASK_PROMPTS[task]["labels"], chat)
    explicit_positions = LLM_MODELS[model].get("model_class") != "image_text_to_text"
    if not all(len(q) == 1 for q in label_seqs):
        n_tokens = n_tokens * len(label_seqs)
    t0 = time.time()
    logits, mass, nonfinite = score(net, tok, encoded, label_seqs, explicit_positions, log)
    seconds = time.time() - t0
    if nonfinite:
        log(f"ABORT {model}: {nonfinite} of {len(rows)} rows with non-finite logits in {job['job_id']}")
        return "nonfinite", tps
    tps = 0.5 * tps + 0.5 * n_tokens / max(seconds, 1e-6)
    out = out_root / job["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    name = job["name"]
    np.save(out / f"{name}_logits.npy", logits)
    np.save(out / f"{name}_labels.npy", labels)
    np.save(out / f"{name}_indices.npy", rows.astype(np.int64))
    np.save(out / f"{name}_label_mass.npy", mass)
    cfg = run_config(job, model, source, revision, tok, label_seqs, demos, cap)
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    acc = float((logits.argmax(1) == labels).mean())
    marker = {"job_id": job["job_id"], "n_items": int(len(rows)), "n_tokens": n_tokens, "seconds": seconds,
              "tokens_per_second": n_tokens / max(seconds, 1e-6), "accuracy": acc,
              "mean_label_mass": float(np.nanmean(mass)), "completed": datetime.now(timezone.utc).isoformat()}
    (out / f"{name}_complete.json").write_text(json.dumps(marker, indent=2))
    files = {f"{name}_{s}.npy": out / f"{name}_{s}.npy" for s in ("logits", "labels", "indices", "label_mass")}
    files["config.json"] = out / "config.json"
    if job["part"] == "target_demonstration":
        (out / f"{name}_demonstrations.json").write_text(json.dumps(demos, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
        files[f"{name}_demonstrations.json"] = out / f"{name}_demonstrations.json"
    files[f"{name}_complete.json"] = out / f"{name}_complete.json"
    ok = store.commit(job, files)
    log(f"done {job['job_id']}: n {len(rows)}, {n_tokens} tokens, {seconds / 60:.1f} min "
        f"({n_tokens / max(seconds, 1e-6):.0f} tok/s), acc {acc:.3f}, label mass {np.nanmean(mass):.3f}, "
        f"upload {'ok' if ok else 'FAILED'}")
    return "done", tps


def worker(args) -> None:
    import gc
    import traceback

    import torch

    tag = f"[worker {args.worker_index}]"
    log = lambda msg: print(f"{datetime.now():%H:%M:%S} {tag} {msg}", flush=True)
    jobs = json.loads(Path(args.phase_file).read_text())
    store = make_store(args.store)
    local_paths = dict(p.split("=", 1) for p in args.model_path)
    n_gpus = torch.cuda.device_count()
    claims = Path(args.phase_file).parent / "claims"
    claims.mkdir(exist_ok=True)
    current, net, tok, source, revision, tps = None, None, None, None, None, None
    for job in jobs:
        model = job["model"]
        if (claims / f"{model}.failed").exists():
            log(f"skip {job['job_id']} ({model} failed earlier)")
            continue
        claim = claims / f"{job['job_id']}.claim"
        try:
            os.close(os.open(claim, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        if store.done(job):
            if not (args.rescore_mismatched_subsets and subset_mismatch(store, job, args.cap)):
                log(f"skip {job['job_id']} (already complete)")
                continue
            log(f"rescore {job['job_id']} (stored item subset differs from the current one)")
        if current != model:
            net = tok = None
            gc.collect()
            torch.cuda.empty_cache()
            try:
                t0 = time.time()
                source = resolve_source(model, local_paths)
                net, tok = load_model(model, source, n_gpus)
                revision = model_revision(model)
                current, tps = model, float(LLM_MODELS[model]["tokens_per_second"])
                log(f"loaded {model} from {source} on {n_gpus} GPU(s) in {time.time() - t0:.0f} s")
            except Exception:
                log(f"ABORT {model}: loading failed\n{traceback.format_exc()[-2000:]}")
                (claims / f"{model}.failed").write_text("load")
                continue
        demos = []
        if job["setting"].startswith("few_shot_target"):
            demos = target_demonstrations(job["task"], job["split"], int(job["setting"].rsplit("seed", 1)[1]), PER_CLASS)
        elif job["setting"].startswith("few_shot"):
            demos = demonstrations(job["task"], int(job["setting"].rsplit("seed", 1)[1]), PER_CLASS)
        try:
            status, tps = run_job(job, net, tok, model, source, revision, demos, args.cap, store, WORK_DIR, tps,
                                  args.deadline, log)
        except Exception:
            log(f"Error in {job['job_id']}\n{traceback.format_exc()[-2000:]}")
            continue
        if status == "deadline":
            os.remove(claim)
            break
        if status == "nonfinite":
            (claims / f"{model}.failed").write_text(job["job_id"])


def prefetch(model: str, source: str) -> bool:
    if Path(source).exists():
        return True
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(source, token=os.environ.get("HF_TOKEN"),
                          allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.py"])
        return True
    except Exception as e:
        print(f"{datetime.now():%H:%M:%S} cannot download {model} ({source}): {type(e).__name__}: {str(e)[:300]}",
              flush=True)
        return False


def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except Exception:
        return 0


def launcher(args) -> None:
    t_start = time.time()
    deadline = t_start + args.deadline_hours * 3600
    jobs = planned_jobs(args)
    print(f"{len(jobs)} jobs for models {args.models}; deadline in {args.deadline_hours} h", flush=True)
    local_paths = dict(p.split("=", 1) for p in args.model_path)
    phases = []
    for job in jobs:
        if phases and phases[-1][0] == job["model"]:
            phases[-1][1].append(job)
        else:
            phases.append((job["model"], [job]))
    n_gpus = gpu_count()
    session = WORK_DIR / "_session"
    session.mkdir(parents=True, exist_ok=True)
    unavailable = set()
    for p_i, (model, phase_jobs) in enumerate(phases):
        if time.time() > deadline - 600:
            print("deadline reached; no further phases", flush=True)
            break
        if model in unavailable or (session / "claims" / f"{model}.failed").exists():
            continue
        if not prefetch(model, resolve_source(model, local_paths)):
            unavailable.add(model)
            continue
        per_worker = LLM_MODELS[model]["gpus"]
        groups = ([[str(g) for g in range(k, k + per_worker)] for k in range(0, n_gpus - per_worker + 1, per_worker)]
                  if n_gpus >= per_worker else [[str(g) for g in range(n_gpus)]] if n_gpus else
                  [[] for _ in range(args.cpu_workers)])
        phase_file = session / f"phase{p_i:02d}.json"
        phase_file.write_text(json.dumps(phase_jobs))
        print(f"{datetime.now():%H:%M:%S} phase {p_i}: {model}, {len(phase_jobs)} jobs, workers on GPUs {groups}",
              flush=True)
        procs = []
        for w, gpus in enumerate(groups):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus), TOKENIZERS_PARALLELISM="false")
            cmd = [sys.executable, "-m", "src.llm_inference", "--worker", "--phase-file", str(phase_file),
                   "--worker-index", str(w), "--store", args.store, "--cap", str(args.cap),
                   "--deadline", str(deadline), *sum([["--model-path", m] for m in args.model_path], []),
                   *(["--rescore-mismatched-subsets"] if args.rescore_mismatched_subsets else []),
                   "--output-prefix", HUB_PREFIX, *(["--target-demonstrations"] if args.target_demonstrations else [])]
            log_path = session / f"phase{p_i:02d}_worker{w}.log"
            procs.append((subprocess.Popen(cmd, env=env, cwd=BASE_DIR, stdout=open(log_path, "w"),
                                           stderr=subprocess.STDOUT), log_path))
        offsets = {str(p): 0 for _, p in procs}
        while True:
            alive = [p.poll() is None for p, _ in procs]
            for _, path in procs:
                text = path.read_text(errors="replace")
                new = text[offsets[str(path)]:]
                offsets[str(path)] = len(text)
                for line in new.splitlines():
                    if any(k in line for k in ("done ", "loaded ", "skip ", "rescore ", "stop ", "ABORT", "RETRY", "Error",
                                               "error", "Traceback", "out of memory")):
                        print(line, flush=True)
            if not any(alive):
                break
            time.sleep(args.poll)
        for p, path in procs:
            if p.returncode:
                print(f"worker exited with code {p.returncode}; last lines of {path.name}:", flush=True)
                print("\n".join(path.read_text(errors="replace").splitlines()[-25:]), flush=True)
    print(f"finished after {(time.time() - t_start) / 3600:.2f} h", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Zero- and few-shot LLM classification on every evaluation split.")
    parser.add_argument("--models", nargs="+", default=["qwen3_8b"], choices=sorted(LLM_MODELS))
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--few-shot-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--cap", type=int, default=CAP)
    parser.add_argument("--skip-few-shot-remainders", action="store_true",
                        help="score few-shot prompts on the primary subsets only")
    parser.add_argument("--splits", nargs="+", default=None, help="score only these splits")
    parser.add_argument("--no-remainders", action="store_true", help="score only the fixed item subsets")
    parser.add_argument("--rescore-mismatched-subsets", action="store_true",
                        help="rescore completed jobs whose stored item indices differ from the current subset")
    parser.add_argument("--deadline-hours", type=float, default=11.0)
    parser.add_argument("--store", default="hub", help="'hub' or 'local:<dir>' (dry runs)")
    parser.add_argument("--model-path", action="append", default=[], help="model_key=local_dir (dry runs)")
    parser.add_argument("--cpu-workers", type=int, default=2)
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--show-plan", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--phase-file", help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--prompt-variants", nargs="+", default=None,
                        help="zero-shot prompt variants (written under llm_prompt_variants/)")
    parser.add_argument("--target-demonstrations", action="store_true",
                        help="few-shot prompts with labelled target demonstrations (written under "
                             "llm_target_demonstrations/; --few-shot-seeds selects the label resamples)")
    parser.add_argument("--output-prefix", default=None, choices=OUTPUT_PREFIXES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    set_output_prefix(args.output_prefix or ("llm_target_demonstrations" if args.target_demonstrations else
                                             "llm_prompt_variants" if args.prompt_variants else "llm_runs"))
    if args.worker:
        worker(args)
    elif args.show_plan:
        for j in planned_jobs(args):
            print(j["job_id"])
    else:
        launcher(args)


if __name__ == "__main__":
    main()
