"""Map every file of the supplementary archive to the part of the manuscript it supports.

Each file shipped by src/build_supplementary_archive.py gets a role and the manuscript sections it supports:
  manuscript        the section sources and the bibliography
  claim table       a results table that tests/test_manuscript_numbers.py reads to check a number in the text;
                    supports = the sections whose claims read it (recorded by executing the claim ledger)
  generated table   a LaTeX table built by src/publication_tables.py and included in a section
  figure            a publication figure included in a section (PDF, with its PNG preview)
  display source    a results table that only feeds a figure or generated table of the manuscript
  theory            the proofs and derivations behind the propositions
  code, test, data documentation, environment
                    the pipeline that rebuilds the tables, its tests, the dataset documentation and the software
                    environment
  thesis analysis   a table or figure that no claim of the manuscript uses (kept for the thesis and for inspection)

    python -m src.supplementary_file_map          writes paper/supplementary_file_map.csv
    python -m src.supplementary_file_map --check  exits with an error if the committed map is out of date
"""

import argparse
import csv
import functools
import io
import re
import sys

from src.supplementary_selection import tracked_files
from src.config import BASE_DIR

MAP_PATH = BASE_DIR / "paper" / "supplementary_file_map.csv"
SECTIONS = BASE_DIR / "paper" / "sections"
LEDGER = BASE_DIR / "tests" / "test_manuscript_numbers.py"
TABLE_ROOTS = {False: "results/tables/grid_max_length_512/", True: "results/tables/"}
NO_CLAIM = "thesis analysis, not used for a claim"
FIELDS = ["path", "role", "supports"]


def ledger_table_use() -> dict:
    """Execute the claim ledger with its table loader instrumented, and return {table path: sections}."""
    accesses = []
    namespace = {"__name__": "claim_ledger", "__file__": str(LEDGER)}
    original = functools.lru_cache

    def recording_cache(*args, **kwargs):
        decorator = original(*args, **kwargs)

        def wrap(func):
            cached = decorator(func) if callable(func) else func
            if getattr(func, "__name__", "") != "table":
                return cached

            @functools.wraps(func)
            def logged(name, top=False):
                accesses.append((len(namespace.get("CLAIMS", [])), TABLE_ROOTS[bool(top)] + f"{name}.csv"))
                return cached(name, top)
            return logged
        return wrap

    sys.path.insert(0, str(BASE_DIR))
    source = LEDGER.read_text(encoding="utf-8")
    assert source.count("from functools import lru_cache\n") == 1
    namespace["lru_cache"] = recording_cache
    exec(compile(source.replace("from functools import lru_cache\n", "\n"), str(LEDGER), "exec"), namespace)
    claims = namespace["CLAIMS"]
    use = {}
    for index, path in accesses:
        if index < len(claims):
            use.setdefault(path, set()).add(claims[index][0])
    data_reads = re.findall(r'ROOT / "data" / "([^"]+\.csv)"', LEDGER.read_text(encoding="utf-8"))
    for name in data_reads:
        use.setdefault(f"data/{name}", set()).add("data_protocol")
    json_reads = re.findall(r'TABLES / "([^"]+\.json)"', LEDGER.read_text(encoding="utf-8"))
    for name in json_reads:
        use.setdefault(TABLE_ROOTS[False] + name, set()).add("statements")
    return use


def builder_sources(module: str, registry: str) -> dict:
    """{output stem: csv paths} for the figure and table builders, from their source code."""
    import ast
    tree = ast.parse((BASE_DIR / "src" / f"{module}.py").read_text(encoding="utf-8"))
    functions = {f.name: f for f in tree.body if isinstance(f, ast.FunctionDef)}

    def reads(name: str, seen: set) -> set:
        if name in seen or name not in functions:
            return set()
        seen.add(name)
        found = set()
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith(".csv"):
                found.add(TABLE_ROOTS[False] + node.value)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                found |= reads(node.func.id, seen)
        return found

    def outputs(name: str) -> list:
        names = [c.args[1].value for c in ast.walk(functions[name]) if isinstance(c, ast.Call)
                 and getattr(c.func, "id", "") == "save" and len(c.args) > 1 and isinstance(c.args[1], ast.Constant)]
        return names or [name.removesuffix("_table")]

    mapping = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == registry:
            for key, value in zip(node.value.keys, node.value.values):
                for out in ([key.value] if registry == "TABLE_BUILDERS" else outputs(value.id)):
                    mapping[out] = reads(value.id, set())
    return mapping


def section_inputs() -> dict:
    """{included file stem: sections} for \\input tables and \\includegraphics figures."""
    found = {}
    for path in sorted(SECTIONS.glob("*.tex")):
        text = path.read_text(encoding="utf-8")
        for stem in re.findall(r"\\input\{\.\./thesis/tables/([^}]+)\}", text):
            found.setdefault(("table", stem), set()).add(path.stem)
        for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
            found.setdefault(("figure", name.rsplit(".", 1)[0]), set()).add(path.stem)
    return found


def displayed_sources(inputs: dict) -> dict:
    """{csv path: sections} for the tables behind every figure and generated table that a section includes."""
    found = {}
    for kind, module, registry in (("figure", "publication_figures", "FIGURES"),
                                   ("table", "publication_tables", "TABLE_BUILDERS")):
        for stem, csvs in builder_sources(module, registry).items():
            for path in csvs:
                found.setdefault(path, set()).update(inputs.get((kind, stem), set()))
    return {k: v for k, v in found.items() if v}


def classify(path: str, use: dict, inputs: dict) -> tuple:
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if path.startswith("paper/sections/") or path == "paper/references.bib":
        return "manuscript", "all sections"
    if path == "paper/supplementary_file_map.csv":
        return "manuscript", "this map"
    if path == "paper/reference_verification_log.csv":
        return "manuscript", "references (how each cited work was verified)"
    if path in use:
        return "claim table", "; ".join(sorted(use[path]))
    if path.startswith("thesis/tables/"):
        sections = inputs.get(("table", stem))
        return ("generated table", "; ".join(sorted(sections))) if sections else ("thesis analysis", NO_CLAIM)
    if path.startswith("thesis/figures/"):
        sections = inputs.get(("figure", stem)) if "/publication/" in path else None
        return ("figure", "; ".join(sorted(sections))) if sections else ("thesis analysis", NO_CLAIM)
    if path.startswith("thesis/theory/"):
        return "theory", "background; shift_triage; appendix"
    if path in inputs.get("displayed", {}):
        return "display source", "; ".join(sorted(inputs["displayed"][path]))
    if path.startswith("results/lora_lr_sweep"):
        return "data documentation", "appendix (LoRA learning-rate selection)"
    if path.startswith("results/"):
        return "thesis analysis", NO_CLAIM
    if path.startswith("src/"):
        return "code", "reproduction of the tables and figures (REPRODUCIBILITY.md)"
    if path.startswith("tests/"):
        return "test", "checks of the code and of every number in the text"
    if path.startswith("data/") or path.startswith("notebooks/"):
        return "data documentation", "data_protocol; appendix"
    return "environment", "reproduction (REPRODUCIBILITY.md)"


def build_rows() -> list:
    use, inputs = ledger_table_use(), section_inputs()
    inputs["displayed"] = displayed_sources(inputs)
    return [dict(zip(FIELDS, (path, *classify(path, use, inputs)))) for path in sorted(tracked_files())]


def render(rows: list) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Map the supplementary files to the claims they support.")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render(build_rows())
    if args.check:
        if not MAP_PATH.exists() or MAP_PATH.read_text(encoding="utf-8") != text:
            raise SystemExit(f"{MAP_PATH.relative_to(BASE_DIR)} is out of date; rerun python -m src.supplementary_file_map")
        print("supplementary file map is up to date")
        return
    MAP_PATH.write_text(text, encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    roles = {}
    for row in rows:
        roles[row["role"]] = roles.get(row["role"], 0) + 1
    print(f"Wrote {MAP_PATH.relative_to(BASE_DIR)}: {len(rows)} files, " +
          ", ".join(f"{k} {v}" for k, v in sorted(roles.items())))


if __name__ == "__main__":
    main()
