import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with open(ROOT / path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def bib_keys():
    text = (ROOT / "paper" / "references.bib").read_text(encoding="utf-8")
    return re.findall(r"^@\w+\{([^,]+),", text, re.M)


def test_every_resolved_reference_is_in_the_bibliography():
    keys = set(bib_keys())
    resolved = [r["key"] for r in rows("paper/reference_verification_log.csv") if r["status"].startswith("resolved")]
    assert resolved and not [k for k in resolved if k not in keys]


def test_unresolved_references_are_not_in_the_bibliography():
    keys = " ".join(bib_keys()).lower()
    for r in rows("paper/reference_verification_log.csv"):
        if r["status"] == "unresolved":
            assert r["use"] == "dropped"
            assert r["key"].split()[0].lower() not in keys


def test_bibliography_keys_are_unique():
    keys = bib_keys()
    assert len(keys) == len(set(keys))


def test_every_corpus_has_a_dated_licence_statement_and_no_text_is_redistributed():
    table = rows("data/corpus_licences.csv")
    assert len(table) == 10
    for r in table:
        assert r["licence_as_stated"] and r["source_page"]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", r["checked_on"])
        assert r["text_redistributed_here"] == "no"


def test_data_sources_names_every_licensed_corpus():
    text = (ROOT / "data" / "DATA_SOURCES.md").read_text(encoding="utf-8")
    for name in ("BD-SHS", "SentNoB", "OLID", "SentiMix", "BIDWESH", "OffMix-3L", "SentMix-3L", "Chaos", "MixTox-SA", "Sylheti"):
        assert name in text
    assert "CC BY-NC-ND" not in text
