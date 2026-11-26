# Progress report

CoRL-template (16-831 course style) progress report. `main.pdf` is the compiled version.

| File | What it is |
| --- | --- |
| `main.tex`, `references.bib` | report source and bibliography |
| `corl_2023.sty`, `corlabbrvnat.bst` | course CoRL template files |
| `scripts/make_report_results.py` | runs every experiment in the report, writes `figures/` and `results.json` |
| `figures/`, `results.json` | generated figures and the raw numbers behind every table |

Regenerate results (from this folder, with the project installed):

```bash
python scripts/make_report_results.py
```

The table values in `main.tex` are typed in by hand from `results.json`, so update
them after rerunning. Then build the PDF:

```bash
latexmk -pdf main.tex
```

Open items are marked in red `[TODO: ...]` in the PDF: the PPO hover results and the
timeline / division of work.
