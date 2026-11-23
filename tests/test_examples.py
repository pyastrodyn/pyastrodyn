"""The worked examples run without error, and their notebooks match the scripts."""

import os
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted((ROOT / "examples").glob("example_*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_example_runs(script, monkeypatch):
    pytest.importorskip("matplotlib")
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: plt.close("all"))
    runpy.run_path(str(script), run_name="__main__")


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_notebook_matches_script(script):
    nbformat = pytest.importorskip("nbformat")
    sys.path.insert(0, str(ROOT / "tools"))
    from make_notebooks import cells_of
    notebook = nbformat.read(script.with_suffix(".ipynb"), as_version=4)
    expected = [cell.source for cell in cells_of(script.read_text())]
    assert [cell.source for cell in notebook.cells] == expected, "run: python tools/make_notebooks.py"
