# ResearchPaper

IEEE journal-format paper (IEEEtran) on genuinely deployable RL battery control (CityLearn 2022). Only real RL methods
are reported (PPO / SAC with different rewards); no perfect-foresight plans and no imitation/cloning results.

| File | What it is |
|---|---|
| `main.tex` | The paper |
| `references.bib` | The 32 references (all cited) |
| `main.pdf` | Compiled paper |
| `figures/make_figures.py` | Rebuilds the three figures from `../results` |

**Rebuild**

1. Figures: `..\venv\Scripts\python.exe figures\make_figures.py`
2. PDF (TinyTeX, installed via `quarto install tinytex`), from this folder:
   `pdflatex main`, `bibtex main`, `pdflatex main`, `pdflatex main`
   (`%APPDATA%\TinyTeX\bin\windows\` holds the executables).

**Before submission:** fill in the affiliation / e-mail placeholder in the `\thanks{}` of `main.tex`, and update the
results if the Round 22 RL runs (TQC, CrossQ, history features, ...) beat M27.
