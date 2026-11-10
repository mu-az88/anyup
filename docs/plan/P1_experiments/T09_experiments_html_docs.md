# T09 — Interactive HTML Documentation of Exp 0–3

| Field | Value |
|---|---|
| Task ID | T09 |
| Plan | P1 Experiments (`plan.md`), Phase B wrap-up |
| Branch | `experiments-task02` |
| Status | Q&A |
| Depends on | T05, T06, T07, T08 (the four experiment notebooks; T08 treated as done, per Monzir) |
| Blocks | — |

## Goal

One self-contained HTML page that documents Exp 0–3. For each experiment it explains what is measured, the config knobs, each notebook step with its code, and the shared `anyup.*` helpers the step calls. A generator script builds the page from the notebooks and the source code, so the page can be rebuilt instead of going stale.

## Scope

- In scope:
  - `docs/experiments/build_docs.py` (generator) → `docs/experiments/experiments.html` (output, committed).
  - Content per experiment as in Q7; a "Shared modules" section for the helpers.
  - Results read from summary JSONs copied into `docs/experiments/results/` when present; otherwise "pending" (Q8).
- Out of scope: changing the notebooks or `anyup/*.py`; the per-clip results table and click-through diagrams (Q9); publishing as a claude.ai Artifact (Q3); `results/*.ipynb` (thesis notebooks); Phase B interpretation of the numbers.

## Q&A (decisions agreed before implementation)

Monzir: "lets go in this branch and the recommended from you options". Q2, Q6, Q7, Q8 and Q9 had no recommendation in the question round, so the answers below marked **(proposed)** are my picks, **to be confirmed with this file**.

| # | Question | Answer |
|---|---|---|
| 1 | Task ID? | **T09** (T08 treated as done). |
| 2 | Audience? | **(proposed) Both internal (Monzir, supervisor) and reviewers**: formal, concise prose; no internal chat history in the main text. Task IDs (T05, S02, …) only in a small "provenance" line per section. |
| 3 | Where? | **Static HTML in the repo** (`docs/experiments/`); opens locally, works on GitHub Pages. |
| 4 | Structure? | **One self-contained file**: sidebar with Overview, Exp 0, Exp 1, Exp 2, Exp 3, Shared modules. No external JS/CSS (works offline). |
| 5 | Sync? | **Generator script** reads the `.ipynb` cells (by their `# ── Cell N · Title` header) and the helpers' signatures + docstrings (`ast`, no import of torch). |
| 6 | Helper links? | **(proposed) GitHub links pinned to the commit hash** at build time (`git rev-parse HEAD`), with line anchors (`#L12-L40`), plus the signature + docstring inline. The build refuses (or warns, flag `--allow-dirty`) if `anyup/` or `experiments/` have uncommitted changes, since the links would point to different lines. |
| 7 | Content per experiment? | **(proposed)** (a) purpose + what is measured; (b) config table of the compute-cost knobs, with the notebook comments; (c) one step per notebook cell with collapsible code; (d) a text data-flow line with tensor shapes (e.g. `(1,1568,768) → (8,768,14,14) → (8,768,224,224)`); (f) results; (g) known issues + S-IDs. (e) protocol history is **short**: one "Protocol vs thesis" box per experiment listing the changes (from `plan.md` Phase B notes), not the full legacy story. |
| 8 | Results? | **(proposed) Read summary JSONs if present, else "pending"**. Monzir copies `exp*_summary.json` from Drive into `docs/experiments/results/` (small files). Thesis values shown next to them (from `plan.md`), labeled as such. |
| 9 | Interactivity? | **(proposed) Light**: sidebar navigation, collapsible code, copy buttons, syntax highlighting (Pygments at build time, so no JS library), light/dark theme. |
| 10 | Branch? | **`experiments-task02`** (current). |

## Design

### Files

- New: `docs/experiments/build_docs.py`, `docs/experiments/experiments.html` (generated), `docs/experiments/results/` (Monzir's JSON copies; `.gitkeep`).
- Hand-written text (purpose, data flow, issues, protocol box) lives in the script as one dict per experiment, so the generated parts (code, signatures, links, results) and the written parts sit in one place.
- No change to code or notebooks.

### Generator

```
python docs/experiments/build_docs.py [--allow-dirty]   # → docs/experiments/experiments.html
```

1. `git rev-parse HEAD` + dirty check (`anyup/`, `experiments/`).
2. Per notebook: split code cells by the `# ── Cell N · Title` header → steps; the config cell is also parsed into a knob table (name, value, trailing comment).
3. Helper usage per step: names imported from `anyup.*` (as found: Exp 0 `metrics`, `BackboneOutput`, `UpsamplerLoader`, `set_seed`; Exp 1 + `ade20k`, `BackboneLoader`, `IMAGENET_*`; Exp 2/3 + `davis`, `TensorSource`, `iter_backbone_chunks`) and attribute calls (`metrics.temporal_cosine`, …) matched against the definitions found by `ast` in `anyup/*.py`, `anyup/utils/seed.py`.
4. Shared-modules section: only the helpers actually used, each with signature, docstring, file:line, GitHub link, and "used by Exp …".
5. Results: `docs/experiments/results/exp{0,1,2,3}_summary.json` if present → table; else "pending".
6. Pygments `HtmlFormatter` → inline CSS; one HTML string written to disk. Deterministic output (sorted, no timestamps except the commit hash).

### Page

- Sidebar: Overview · Exp 0 Efficiency · Exp 1 ADE20K probe · Exp 2 Temporal consistency · Exp 3 Label propagation · Shared modules.
- Overview: the four experiments in one table (what, backbone, data, metric, conditions), the shared pipeline (backbone → upsampler → metric), reproducibility notes (seed, CUBLAS, env/hash in results).
- Per experiment: purpose · data flow with shapes · config knobs · steps (cell title, short description, helper chips linking to Shared modules, collapsible code) · results vs thesis · protocol vs thesis · known issues.

## Steps

1. Generator skeleton: notebook parsing + helper index; print what was found (cells per notebook, helpers per step) for Monzir to check.
2. HTML template + styling; generate a first page.
3. Hand-written text per experiment (purpose, shapes, protocol box, issues), facts checked against the code/task files.
4. Results loading + "pending" fallback.
5. Monzir reviews the page in the browser; fixes.

## Tests / Regression Checks

- Every code cell of the four notebooks appears exactly once in the page (count check printed by the script).
- Every helper link resolves to a `def`/`class` line at the pinned commit (script checks the line content via `git show <hash>:<file>`).
- Every shape and setting in the hand-written text is checked against the code (listed in the Implementation Log).
- Rendering check in a local browser (desktop + phone width), light and dark.

## Done When

- `build_docs.py` runs on a clean tree and writes `experiments.html` with all checks passing.
- Monzir has reviewed the page.

## Implementation Log

_Updated live during implementation. Per step: what changed, files touched, test results (actual numbers/shapes). Also log mid-implementation questions and answers, changed decisions (old → new, why), and anything added or removed. Sections above are edited in place to match, so the finished file is a report of what was actually done._
