# T09 — Interactive HTML Documentation of Exp 0–3

| Field | Value |
|---|---|
| Task ID | T09 |
| Plan | P1 Experiments (`plan.md`), Phase B wrap-up |
| Branch | `experiments-task02` |
| Status | done |
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

Monzir: "lets go in this branch and the recommended from you options". Q2, Q6, Q7, Q8 and Q9 had no recommendation in the question round; the answers marked **(proposed)** were my picks and were accepted with the file ("yes lets do it").

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

- New: `docs/experiments/build_docs.py` (mechanics), `docs/experiments/doc_content.py` (hand-written text), `docs/experiments/experiments.html` (generated), `docs/experiments/results/` (Monzir's copies of the summary JSONs + `exp0_per_item.csv`; `.gitkeep`).
- Hand-written text (purpose, data flow, step descriptions, implicit helpers, protocol box, thesis values, issues) lives in `doc_content.py`, one dict per experiment; everything else is generated.
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

- **OK received ("yes lets do it")**, proposed answers Q2/Q6–Q9 accepted. Status Q&A → in progress.
- **Step 1 (code): `docs/experiments/build_docs.py`**, parser + helper index + `--report` (HTML step not yet implemented; without `--report` it exits with a message). Standard library only, no anyup import.
  - Notebooks: markdown intro + code cells split by the `# ── Cell N · Title` header (error if a cell has no header or numbers are not 1..N). Config cell → knobs (UPPER top-level assignments, value, inline + indented continuation comments, `↓` = compute-cost knob).
  - Helper index (`ast`): 168 definitions (top-level functions/classes/UPPER constants + methods) in `anyup/{backbones,video,upsamplers,davis,metrics,ade20k}.py`, `anyup/utils/seed.py`.
  - Helper detection per cell: `module.attr` on imported anyup modules, imported names, `Class.method` on imported classes, and `obj.method` when the method name belongs to one anyup class only (generic names like `load`, `update`, `result`, `forward` excluded). A bare class is dropped when one of its methods is listed (`UpsamplerLoader.load` covers `UpsamplerLoader`).
  - `--report` at commit `13fce8c` (clean `anyup/`, `experiments/`): Exp 0 7 cells / 22 config assignments (4 ↓), Exp 1 7 / 34 (8 ↓), Exp 2 6 / 15 (4 ↓), Exp 3 6 / 20 (5 ↓); **34 shared helpers used**. `save_results` used by all 4, `set_seed` / `UpsamplerLoader.load` by all 4, `paired_stats` by Exp 1–3.
  - Found: `TensorSource` imported but unused in Exp 2 and Exp 3; `BackboneOutput` imported but unused in Exp 3 (notebooks not changed, out of scope; shown as INFO in the report only).
  - Limitation: calls through objects are not detected statically: `UPS[c](ch)` (→ `Upsampler.forward` / `_run` and each subclass's `_upsample`), the backbone call inside `iter_backbone_chunks`, `stream.update` / `.result`. Plan: a small hand-written "implicit helpers" list per step, kept in the script with the other hand-written text (Step 3).
- **Answers after Step 1 (Monzir: "do the recommended"):** (a) a hand-written list of implicit helpers per step; (b) the config table hides derived names. Rule used: an assignment is hidden when its value references an earlier config name **and** it has no comment (hidden: Exp 0 `GPU, P, RUN_NAME, OUT_DIR`; Exp 1 the same + `CKPT_TAG, DATA_TAG, CACHE_TAG, HEAD_DIR, CACHE_DRIVE, CACHE_LOCAL`; Exp 2/3 none).
- **Steps 2–4 (code):**
  - Changed from the design (old → new): hand-written text in a separate `docs/experiments/doc_content.py` instead of inside the script (why: the text is what will be edited most; the mechanics stay untouched). Design → Files updated.
  - Added to the index: `anyup/model.py`, `anyup/model_2d.py` (so `AnyUp.forward` / `AnyUp2D.forward` can be linked as implicit helpers); 178 definitions in 9 files. Added a duplicate-key guard (`Class.method` across files).
  - `doc_content.py`: overview + table, reproducibility, and per experiment: purpose, data flow with shapes, one description per cell (the build errors if a cell has none), implicit helpers (Exp 0 Cell 4; Exp 1 Cells 3, 4; Exp 2 Cell 4; Exp 3 Cell 4: backbone `forward` + adapter `_extract`, `Upsampler.forward`, each `_upsample`/`_upsample_images`, `AnyUp.forward`, `AnyUp2D.forward`, `TemporalCosineStream.update/.result`), protocol vs thesis, thesis values, issues.
  - Facts checked against the code: VideoMAE (t ≤ 8, 768, 14, 14), guide (t, 3, 224, 224), frames 0, 2, 4, … kept with overlap (`iter_backbone_chunks`); Swin-T per frame 224 → (t, 768, 7, 7) (`patch_size=32`), guide at `PROP_SIZE`; DINOv2-base 224 → (B, 768, 16, 16); Exp 1 head `Conv2d(768, 150, 1)`; temporal cosine per pixel along C, mean over pixels then pairs (`temporal_cosine`); propagation context frame 0 + last 6 (`N_LAST`); Exp 0 bilinear memory check tol 0.10. Thesis Exp 1 mIoU (39.54 / 42.36 / 41.93, step500) taken from S02; the rest from `plan.md`.
  - Added to the issues (not in the plan): **S04** on Exp 1–3 (regression checks ran before T07's deterministic seeding); the T05 regression statement is phrased as "on the old protocol".
  - Renderer: one self-contained HTML (inline CSS/JS, Pygments `default` / `github-dark` at build time, no external requests), sidebar with active-section highlight, light/dark (system default + toggle, saved per browser), collapsible code with copy button, helper chips (solid = found by the parser, dashed = implicit) → "Shared modules" cards (signature, docstring, `file:lines` GitHub link pinned to the commit, back-links to the cells).
  - Results: reads `docs/experiments/results/exp{0..3}_summary.json` (+ `exp0_per_item.csv` for the main-row table, since times are only in the CSV); otherwise "Pending" with the Drive folder. Thesis reference table always shown.
  - Build: refuses on uncommitted changes in `anyup/` or `experiments/` (`--allow-dirty` adds a warning banner); warns if the commit is not on a remote branch; writes nothing if a check fails.
- **Build run (commit `13fce8c`, on `origin/experiments-task02`):** [PASS] every code cell once: Exp 0 7/7, Exp 1 7/7, Exp 2 6/6, Exp 3 6/6; [PASS] helper links match the pinned commit 48/48 (`git show 13fce8c:<file>`, def/class/constant line); [PASS] every helper chip has a card (48). Results: none present → all four Results sections show Pending. Output 406 KiB.
- **Browser check (Playwright, Chromium, local server):** desktop 1360 px light + dark, phone 390 px: no horizontal page scroll (scrollWidth 375 ≤ 390); code and tables scroll inside their boxes. Fixed after the first look: config Value column too narrow (paths broke mid-word), "0 derived names" sentence shown for Exp 2/3, overview first column wrapping, missing favicon (inline SVG). Remaining console entry: the browser's automatic `/favicon.ico` request to the test server (404), harmless.
- **Step 5, review round 1 (Monzir):** "hyper links need slightly noticeable colors, and add expand and collaps to the menu in the left too, I dont want it to always be there. and make the seperations between the experiments clear in the main page, make a thick divider". Results files copied into `docs/experiments/results/` (exp0 summary + per-item CSV, exp1–3 summaries).
  - Changes (`build_docs.py` CSS/JS/markup only; `doc_content.py` unchanged):
    - Links: new `--link` colour (light `#1f56d6`, dark `#8cb0ff`); links in the content get a soft underline (`--link-line`, solid on hover). Helper chips are now link-coloured on a light blue fill (`--chip-bg`, `--chip-line`); implicit chips stay dashed, also link-coloured.
    - Sidebar: a fixed "Hide menu / ☰ Menu" button (top left). Closed → sidebar hidden, content centred. Default: open on desktop, closed at ≤ 860 px; the choice is remembered per browser (`localStorage`, wrapped in try/catch).
    - Experiments: `section.exp` border 1 px → **6 px divider** (`--divider`, grey in both themes), 72 px above, h2 26 px.
  - Build (commit `13fce8c`): all checks PASS (cells 7/7, 7/7, 6/6, 6/6; helper links 48/48; chip cards 48). All 5 results files found; Exp 0 results render (run `A100_main_s224_f8_q4096_sweep`, Tesla T4). Output 412 KiB.
  - Browser check (Playwright): 1360 px light with menu open, then closed; dark mode with chips; 390 px phone: menu closed by default, no horizontal page scroll (scrollWidth 375).
- **Step 5, review round 2 (Monzir):** "I want collapse/expand for topics under each experiments in the left bar".
  - Changes (`build_docs.py` only): each experiment in the sidebar is a group (experiment link + ▸ button) with its topics (Data flow, Configuration, Steps, Results) in a nested list. The ▸ button shows or hides the topics; the experiment name still jumps to the section. "Expand all" / "Collapse all" buttons above the list. Default: all collapsed; open groups are remembered per browser (`localStorage`, try/catch). The phone layout now shows the same list instead of the old wrapped row with hidden topics.
  - Build (commit `13fce8c`): all checks PASS (cells, 48/48 links, 48 cards); 415 KiB.
  - Browser check: 1360 px: Exp 1 opened via ▸, others closed, state saved (`["exp1"]`); the active-section highlight still follows scrolling (`#exp2` when reading Exp 2); 390 px: menu open shows the groups, no horizontal page scroll (scrollWidth 375).
- **Close-out.** Monzir accepted the page and closed P1 ("consider the phase done"). Status → done. Final page: `docs/experiments/experiments.html`, 415 KiB, built from commit `13fce8c`, all build checks PASS. Rebuild with `python3 docs/experiments/build_docs.py` after any change in `anyup/` or `experiments/` (it refuses on uncommitted changes there). Left over: none for T09; open experiment items are S02, S03, S06–S10.
