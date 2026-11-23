"""Write the notebook of every example script.

The scripts in ``examples/`` are the source: cells are separated by lines
``# %%`` (code) and ``# %% [markdown]`` (text, as comment lines). Run from
the repository folder:

    python tools/make_notebooks.py

Each ``examples/<name>.py`` gets an ``examples/<name>.ipynb`` with the same
content, without results.
"""

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent


def cells_of(script):
    """Split a percent-format script into (kind, text) cells."""
    cells, kind, lines = [], None, []
    for line in script.splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                cells.append((kind, lines))
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        elif kind is not None:
            lines.append(line)
    if kind is not None:
        cells.append((kind, lines))
    notebook_cells = []
    for kind, lines in cells:
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if kind == "markdown":
            text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in lines)
            notebook_cells.append(nbformat.v4.new_markdown_cell(text))
        elif lines:
            notebook_cells.append(nbformat.v4.new_code_cell("\n".join(lines)))
    return notebook_cells


def main():
    for script in sorted((ROOT / "examples").glob("*.py")):
        notebook = nbformat.v4.new_notebook()
        notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}}
        notebook.cells = cells_of(script.read_text())
        for cell in notebook.cells:
            cell.pop("id", None)
        nbformat.write(notebook, script.with_suffix(".ipynb"), version=4)
        print("written:", script.with_suffix(".ipynb").relative_to(ROOT))


if __name__ == "__main__":
    main()
