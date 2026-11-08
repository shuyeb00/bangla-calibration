"""Every file of the supplementary archive is mapped to the manuscript part it supports, or marked as unused."""

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build_supplementary_archive import tracked_files  # noqa: E402
from src import supplementary_file_map as fm  # noqa: E402

pytestmark = pytest.mark.skipif(not (ROOT / ".git").exists(),
                                reason="the file map is built from git ls-files; run inside a git checkout")

ROLES = {"manuscript", "claim table", "display source", "generated table", "figure", "theory", "code", "test",
         "data documentation", "environment", "thesis analysis"}


def committed_map() -> list:
    with open(fm.MAP_PATH, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_every_shipped_file_is_mapped():
    mapped = [row["path"] for row in committed_map()]
    shipped = set(tracked_files())
    assert len(mapped) == len(set(mapped))
    assert not sorted(shipped - set(mapped)), sorted(shipped - set(mapped))
    assert not sorted(set(mapped) - shipped), sorted(set(mapped) - shipped)


def test_roles_and_supports_are_filled():
    for row in committed_map():
        assert row["role"] in ROLES, row
        assert row["supports"], row
        assert (row["role"] == "thesis analysis") == (row["supports"] == fm.NO_CLAIM), row


def test_every_displayed_figure_and_table_is_shipped_and_mapped():
    rows = {row["path"]: row for row in committed_map()}
    for kind, stem in fm.section_inputs():
        path = (f"thesis/tables/{stem}.tex" if kind == "table"
                else f"thesis/figures/grid_max_length_512/publication/{stem}.pdf")
        assert path in rows and rows[path]["role"] in {"generated table", "figure"}, path


def test_map_is_up_to_date():
    assert fm.render(fm.build_rows()) == fm.MAP_PATH.read_text(encoding="utf-8")
