"""Anonymised supplementary archive for double-blind review.

Packs the tracked code, the tables and figures of the submitted study, LaTeX tables, theory notes and the manuscript
sections (whose numbers tests/test_manuscript_numbers.py checks; the title page with the author list is left out) into
a deterministic ZIP. The superseded max_length-128 grid and the repository README stay out, so that the supplement
holds only material that supports the submission. Author names, e-mail addresses, institution, the authors' repository
and Hugging Face identifiers and git commit hashes are replaced; links to third-party dataset repositories are kept,
since the licence table cites them, and the archive is scanned afterwards; the build fails if any identifying string remains or the archive
exceeds the size limit. With --outputs, shards of the released model outputs (src/export_release_outputs.py export
--anonymise) are added in priority order while they fit, with a manifest reduced to the shards included; their JSON
members are scanned as well.

    python -m src.build_supplementary_archive [--outputs release/outputs_anonymised] [--out release/supplementary.zip]
"""

import argparse
import hashlib
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path

from src.config import BASE_DIR
from src.supplementary_selection import tracked_files

DEFAULT_OUT = BASE_DIR / "release" / "supplementary.zip"
LIMIT_BYTES = 100 * 10**6
TEXT_SUFFIXES = {".py", ".md", ".txt", ".csv", ".sh", ".tex", ".json", ".ipynb", ".gitignore", ".cfg", ".toml", ""}
SHARD_PRIORITY = ("runs_offensive", "runs_sentiment", "runs_offensive_olid", "runs_sentiment_hinglish",
                  "llm_runs_offensive", "llm_runs_sentiment", "few_label_finetuning", "llm_prompt_variants_offensive",
                  "llm_prompt_variants_sentiment", "llm_target_demonstrations_offensive",
                  "llm_target_demonstrations_sentiment", "llm_membership_inference")
ZIP_TIME = (1980, 1, 1, 0, 0, 0)

REPLACEMENTS = [
    (re.compile(r"https?://github\.com/(?:shuyeb00|suheb00|FerdusAhmed)[^\s)\"'\\,]*", re.I), "<anonymised URL>"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<anonymised e-mail>"),
    (re.compile(r"\*?Authors:\*?[^\n]*"), "Authors: anonymised for review"),
    (re.compile(r"Abu Jafar Md Jakaria|Md Suheb Ahmed|Suheb Ahmed|Ferdus Ahmed|Ferdous Ahmed", re.I),
     "<anonymised author>"),
    (re.compile(r"FerdusAhmed|shuyeb00|suheb00|Suheb|Shuyeb|Ferdus|Ferdous|Jakaria", re.I), "<anonymised>"),
    (re.compile(r"Metropolitan University", re.I), "<anonymised institution>"),
    (re.compile(r"bangla-calibration-artifacts"), "project-artifacts"),
    (re.compile(r"bangla-calibration/(encoder-checkpoints|run-outputs)"), r"anonymous-org/\1"),
    (re.compile(r"bangla-calibration"), "project"),
]
COMMIT_IN_TEXT = re.compile(r"(?<![0-9a-zA-Z])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{7}(?![0-9a-zA-Z])")
FORBIDDEN = re.compile(r"suheb|shuyeb|ferdus|ferdous|jakaria|metropolitan|gmail|bangla-calibration",
                       re.I)


def anonymise_csv(text: str) -> str:
    lines = text.splitlines(keepends=True)
    if not lines:
        return text
    header = lines[0].rstrip("\r\n").split(",")
    if "code_commit" not in header:
        return text
    j = header.index("code_commit")
    out = [lines[0]]
    for line in lines[1:]:
        end = line[len(line.rstrip("\r\n")):]
        cells = line.rstrip("\r\n").split(",")
        if len(cells) == len(header):
            cells[j] = "redacted"
        out.append(",".join(cells) + end)
    return "".join(out)


def anonymise(name: str, text: str) -> str:
    for pattern, replacement in REPLACEMENTS:
        text = pattern.sub(replacement, text)
    if name.endswith(".csv"):
        text = anonymise_csv(text)
    if Path(name).suffix in (".md", ".py", ".ipynb", ".txt"):
        text = COMMIT_IN_TEXT.sub("redacted", text)
    return text


def is_text(name: str) -> bool:
    return Path(name).suffix in TEXT_SUFFIXES


def reduced_manifest(bundle: Path, shards: list) -> bytes:
    manifest = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))
    files = {k: v for k, v in manifest["files"].items() if v["shard"] in shards}
    labels = [r for r in manifest["labels"] if r["path"].split("::")[0] in files]
    manifest.update({"shards": {k: v for k, v in manifest["shards"].items() if k in shards}, "files": files,
                     "labels": labels})
    manifest["left_out"]["shards_not_in_supplement"] = sorted(set(json.loads(
        (bundle / "MANIFEST.json").read_text(encoding="utf-8"))["shards"]) - set(shards))
    return (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def readme(shards: list, left_out: list) -> bytes:
    text = ["# Supplementary material for \"When Thresholds Move: Bangla Text Classification Under Dialect, Script "
            "and Code-Mixing Shift\"", "", "Anonymised for review.", "",
            "Contents: code (`src/`, `notebooks/00_data_preparation.ipynb`), tests (`tests/`, including the "
            "recomputation of every number in the manuscript text from the tables), the manuscript sections "
            "(`paper/sections/`), the bibliography and its verification log (`paper/`), the map from every file to "
            "the claims it supports (`paper/supplementary_file_map.csv`), derived tables "
            "(`results/tables/grid_max_length_512/` for the study, `results/tables/` for data, tokenizer and "
            "human-validation tables, `results/lora_lr_sweep*/` for the learning-rate selection), figures, LaTeX "
            "tables and theory notes (`thesis/`), dataset documentation and the licence table (`data/`) and the "
            "reproduction script (`reproduce.sh`). Identifying names, links and commit hashes are replaced. See "
            "`REPRODUCIBILITY.md` for the levels of reproduction.", "",
            "Level 1 (tables to figures, LaTeX tables and unit tests, about eight minutes on a CPU, Python 3.11): "
            "`pip install -r requirements-analysis.txt && bash reproduce.sh tables`, or `python -m pytest tests` for "
            "the tests alone. Some tests skip here by design: the title-length test reads the title page "
            "(`paper/main.tex`), which carries the author list and is left out, and the file-map tests list files "
            "with `git ls-files`; the split-size test needs the prepared datasets. "
            "`bash reproduce.sh verify` also needs a git checkout.", ""]
    if shards:
        text += ["Level 2 (model outputs to tables): prepare the datasets with the scripts listed in "
                 "`data/DATA_SOURCES.md`, then run `python -m src.export_release_outputs restore --bundle "
                 "outputs` and `bash reproduce.sh outputs`. The outputs contain logits and item indices but no "
                 "text and no labels; labels are rebuilt from the prepared datasets and checked against SHA-256 "
                 "hashes.", "", "Shards included: " + ", ".join(shards) + "."]
        if left_out:
            text += ["Shards left out for size (in the archived output deposit): " + ", ".join(left_out) + "."]
    else:
        text += ["Level 2 (model outputs to tables): this copy carries no model outputs; the logits and item "
                 "indices are in the archived output deposit named in the manuscript's data statement. Without "
                 "them the study is reproducible here at level 1."]
    return ("\n".join(text) + "\n").encode("utf-8")


def add(z: zipfile.ZipFile, name: str, data: bytes, compress: bool = True) -> None:
    info = zipfile.ZipInfo(name, ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    z.writestr(info, data, compresslevel=9 if compress else None)


def build(out: Path, outputs: Path, limit: int) -> None:
    members = {}
    for f in tracked_files():
        data = (BASE_DIR / f).read_bytes()
        if is_text(f):
            # Windows checkouts carry CRLF, which breaks reproduce.sh under bash on other systems.
            data = anonymise(f, data.decode("utf-8").replace("\r\n", "\n")).encode("utf-8")
        members[f] = data
    base_size = len(zip_bytes(members, {}))
    shards, left_out, extra = [], [], {}
    if outputs:
        available = json.loads((outputs / "MANIFEST.json").read_text(encoding="utf-8"))["shards"]
        order = [f"{s}.tar.xz" for s in SHARD_PRIORITY if f"{s}.tar.xz" in available]
        order += sorted(set(available) - set(order))
        budget = limit - base_size - 2 * 10**6
        for name in order:
            size = available[name]["bytes"]
            if size <= budget:
                shards.append(name)
                budget -= size
            else:
                left_out.append(name)
        for name in shards:
            extra[f"outputs/{name}"] = (outputs / name).read_bytes()
        manifest = reduced_manifest(outputs, shards)
        extra["outputs/MANIFEST.json"] = manifest
        sums = [f"{hashlib.sha256(extra[f'outputs/{n}']).hexdigest()}  {n}" for n in shards]
        sums.append(f"{hashlib.sha256(manifest).hexdigest()}  MANIFEST.json")
        extra["outputs/SHA256SUMS"] = ("\n".join(sums) + "\n").encode()
    members["README.md"] = readme(shards, left_out)
    data = zip_bytes(members, extra)
    leaks = [n for n, d in members.items() if is_text(n) and FORBIDDEN.search(d.decode("utf-8", "replace"))]
    for name, blob in extra.items():
        if name.endswith(".tar.xz"):
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:xz") as tar:
                for m in tar.getmembers():
                    if m.name.endswith((".json", ".jsonl")):
                        text = tar.extractfile(m).read().decode("utf-8", "replace")
                        if FORBIDDEN.search(text) or COMMIT_IN_TEXT.search(text.replace("redacted", "")):
                            leaks.append(f"{name}:{m.name}")
    if leaks:
        raise SystemExit(f"Identifying strings remain in {leaks[:10]}")
    if len(data) > limit:
        raise SystemExit(f"Archive is {len(data) / 1e6:.1f} MB, above the limit of {limit / 1e6:.0f} MB")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"Wrote {out} ({len(data) / 1e6:.1f} MB, {len(members) + len(extra)} files); output shards included: "
          f"{len(shards)}, left out for size: {left_out or 'none'}; scan for identifying strings: clean")


def zip_bytes(members: dict, extra: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name in sorted(members):
            add(z, name, members[name])
        for name in sorted(extra):
            add(z, name, extra[name], compress=not name.endswith(".tar.xz"))
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Build the anonymised supplementary archive.")
    parser.add_argument("--outputs", type=Path, help="directory written by src.export_release_outputs export")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit-mb", type=float, default=LIMIT_BYTES / 1e6)
    args = parser.parse_args()
    build(args.out, args.outputs, int(args.limit_mb * 1e6))


if __name__ == "__main__":
    main()
