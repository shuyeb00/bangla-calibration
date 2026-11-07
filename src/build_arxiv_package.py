"""Self-contained LaTeX source of the manuscript for arXiv, and the two PDFs built from the same source.

The manuscript uses the ACM class (acmart, acmsmall format of ACM TALLIP). The arXiv source uses the `nonacm` option
(no ACM reference block, copyright notice or line numbers), places every figure and generated LaTeX table next to
main.tex, and ships the class file and the compiled bibliography (main.bbl), so arXiv needs no BibTeX run and no file
outside the archive. The same sources with the journal options (`review`: line numbers) give the submission PDF. Both
PDFs are compiled from the extracted archive in an empty directory, which checks that the archive is complete.

    python -m src.build_arxiv_package [--out release]
Writes release/arxiv_source.tar.gz, release/manuscript_preprint.pdf and release/manuscript_submission.pdf.
Needs pdflatex and bibtex on the PATH.
"""

import argparse
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from src.config import BASE_DIR

PAPER = BASE_DIR / "paper"
FIGURES = BASE_DIR / "thesis" / "figures" / "grid_max_length_512" / "publication"
TABLES = BASE_DIR / "thesis" / "tables"
STYLE_FILES = ("acmart.cls", "ACM-Reference-Format.bst", "math_commands.tex", "references.bib")
DOCUMENTCLASS = re.compile(r"\\documentclass\[[^\]]*\]\{acmart\}")
CLASS_OPTIONS = {"preprint": "acmsmall,screen,nonacm", "submission": "acmsmall,screen,review"}
INPUT_TABLE = re.compile(r"\\input\{\.\./thesis/tables/([^}]+)\}")
GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def latex(directory: Path, max_runs: int = 5) -> None:
    def run(cmd):
        out = subprocess.run(cmd, cwd=directory, capture_output=True, text=True)
        if out.returncode != 0 and cmd[0] == "pdflatex":
            log = (directory / "main.log").read_text(errors="replace")
            raise SystemExit(f"{' '.join(cmd)} failed in {directory}:\n" + "\n".join(log.splitlines()[-30:]))
        return out

    def log():
        return (directory / "main.log").read_text(errors="replace")

    run(["pdflatex", "-interaction=nonstopmode", "main"])
    if (directory / "references.bib").exists():
        run(["bibtex", "main"])
    for _ in range(max_runs):
        run(["pdflatex", "-interaction=nonstopmode", "main"])
        if "Rerun to get" not in log():
            break
    problems = [line for line in log().splitlines() if "undefined" in line or "Rerun to get" in line]
    if problems:
        raise SystemExit(f"Unresolved references in {directory}: {problems[:5]}")


def staged_source(stage: Path) -> None:
    """Copy the manuscript into `stage` with every external file placed next to main.tex."""
    (stage / "sections").mkdir(parents=True)
    for name in STYLE_FILES:
        shutil.copy2(PAPER / name, stage / name)
    graphics, tables = set(), set()
    for tex in sorted((PAPER / "sections").glob("*.tex")):
        text = tex.read_text(encoding="utf-8")
        tables.update(INPUT_TABLE.findall(text))
        graphics.update(GRAPHIC.findall(text))
        text = INPUT_TABLE.sub(lambda m: r"\input{tables/" + m.group(1) + "}", text)
        (stage / "sections" / tex.name).write_text(text, encoding="utf-8")
    (stage / "figures").mkdir()
    for g in sorted(graphics):
        shutil.copy2(FIGURES / g, stage / "figures" / g)
    (stage / "tables").mkdir()
    for t in sorted(tables):
        name = t if t.endswith(".tex") else f"{t}.tex"
        shutil.copy2(TABLES / name, stage / "tables" / name)
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    main = re.sub(r"\\graphicspath\{\{[^}]*\}\}", r"\\graphicspath{{figures/}}", main)
    main = "\n".join(line for line in main.splitlines() if not line.lstrip().startswith("%"))
    (stage / "main.tex").write_text("\\pdfoutput=1\n" + main + "\n", encoding="utf-8")


def with_option(main: Path, option: str) -> None:
    text = main.read_text(encoding="utf-8")
    new, n = DOCUMENTCLASS.subn(lambda m: "\\documentclass[" + CLASS_OPTIONS[option] + "]{acmart}", text, count=1)
    if n != 1:
        raise SystemExit("main.tex has no \\documentclass[...]{acmart} line")
    main.write_text(new, encoding="utf-8")


def build(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stage = tmp / "stage"
        staged_source(stage)
        with_option(stage / "main.tex", "preprint")
        latex(stage)
        for name in ("references.bib", "ACM-Reference-Format.bst"):
            (stage / name).unlink()
        for p in stage.iterdir():
            if p.suffix in (".aux", ".log", ".blg", ".out", ".pdf"):
                p.unlink()
        archive = out / "arxiv_source.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    info = tar.gettarinfo(str(p), arcname=str(p.relative_to(stage)))
                    info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
                    with open(p, "rb") as f:
                        tar.addfile(info, f)
        for option, pdf in (("preprint", "manuscript_preprint.pdf"), ("submission", "manuscript_submission.pdf")):
            check = tmp / f"check_{option}"
            check.mkdir()
            with tarfile.open(archive) as tar:
                if hasattr(tarfile, "data_filter"):
                    tar.extractall(check, filter="data")
                else:
                    tar.extractall(check)
            with_option(check / "main.tex", option)
            latex(check)
            shutil.copy2(check / "main.pdf", out / pdf)
    print(f"Wrote {archive} ({archive.stat().st_size / 1e6:.2f} MB), {out / 'manuscript_preprint.pdf'} and "
          f"{out / 'manuscript_submission.pdf'}; both compiled from the archive alone")


def main():
    parser = argparse.ArgumentParser(description="arXiv source archive and manuscript PDFs.")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "release")
    args = parser.parse_args()
    build(args.out)


if __name__ == "__main__":
    main()
