"""Build docs/experiments/experiments.html from the experiment notebooks and the anyup source (T09).

    python docs/experiments/build_docs.py --report        # print what was parsed (no HTML)
    python docs/experiments/build_docs.py [--allow-dirty] # write experiments.html

Needs only the standard library (+ Pygments for the HTML step). Nothing from anyup is imported:
notebooks are read as JSON, helpers are found with `ast`.
"""
from __future__ import annotations

import argparse
import csv
import html
import ast
import io
import json
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO / "docs" / "experiments"
OUT_HTML = DOCS_DIR / "experiments.html"
RESULTS_DIR = DOCS_DIR / "results"
GITHUB = "https://github.com/mu-az88/anyup"

NOTEBOOKS = {   # exp id → notebook (order = page order)
    "exp0": "experiments/exp0_efficiency.ipynb",
    "exp1": "experiments/exp1_ade20k_probe.ipynb",
    "exp2": "experiments/exp2_temporal_consistency.ipynb",
    "exp3": "experiments/exp3_label_propagation.ipynb",
}
HELPER_FILES = [   # modules the notebooks import from; module path → file
    ("anyup.backbones", "anyup/backbones.py"),
    ("anyup.video", "anyup/video.py"),
    ("anyup.upsamplers", "anyup/upsamplers.py"),
    ("anyup.davis", "anyup/davis.py"),
    ("anyup.metrics", "anyup/metrics.py"),
    ("anyup.ade20k", "anyup/ade20k.py"),
    ("anyup.utils.seed", "anyup/utils/seed.py"),
    ("anyup.model", "anyup/model.py"),         # AnyUp3D network (reached through AnyUp3DUpsampler)
    ("anyup.model_2d", "anyup/model_2d.py"),   # AnyUp2D network (reached through AnyUp2DUpsampler)
]
CELL_HEADER = re.compile(r"^# ── Cell (\d+) · (.+?)\s*─*\s*$")
# method names too generic to map `obj.name(...)` to one anyup class without type information
GENERIC_METHODS = {"load", "available", "update", "result", "get", "forward", "__init__", "__len__", "__getitem__"}


# ── Helper index ─────────────────────────────────────────────────────────────
@dataclass
class Helper:
    qualname: str          # e.g. "metrics.save_results", "UpsamplerLoader.load"
    module: str            # e.g. "anyup.metrics"
    file: str              # repo-relative path
    kind: str              # function | class | method | constant
    lineno: int            # first line (decorator if any) → link start
    end_lineno: int
    def_lineno: int        # the `def` / `class` / assignment line → link check
    signature: str
    doc: str
    used_by: Dict[str, List[int]] = field(default_factory=dict)   # exp id → cell numbers


def _signature(node) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        return f"{node.name}({ast.unparse(node.args)}){ret}"
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    return ast.unparse(node)


def build_index() -> Dict[str, Helper]:
    """Top-level functions / classes / UPPER constants and class methods of HELPER_FILES.
    Keys: "module_short.name" for module-level names (e.g. "metrics.save_results"), and
    "Class.method" for methods. Module-level names are also reachable by bare name via `by_name`."""
    index: Dict[str, Helper] = {}
    for module, rel in HELPER_FILES:
        short = module.split(".")[-1]
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
                index[f"{short}.{node.name}"] = Helper(f"{short}.{node.name}", module, rel, kind, start,
                                                       node.end_lineno, node.lineno, _signature(node), ast.get_docstring(node) or "")
                if isinstance(node, ast.ClassDef):
                    for sub in node.body:
                        if isinstance(sub, ast.FunctionDef):
                            s0 = sub.decorator_list[0].lineno if sub.decorator_list else sub.lineno
                            if f"{node.name}.{sub.name}" in index:
                                raise KeyError(f"duplicate helper key {node.name}.{sub.name} ({rel})")
                            index[f"{node.name}.{sub.name}"] = Helper(
                                f"{node.name}.{sub.name}", module, rel, "method", s0, sub.end_lineno, sub.lineno,
                                _signature(sub), ast.get_docstring(sub) or "")
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id.isupper():
                name = node.targets[0].id
                index[f"{short}.{name}"] = Helper(f"{short}.{name}", module, rel, "constant", node.lineno,
                                                  node.end_lineno, node.lineno, f"{name} = {ast.unparse(node.value)}", "")
    return index


# ── Notebook parsing ─────────────────────────────────────────────────────────
@dataclass
class Step:
    number: int
    title: str
    code: str
    helpers: List[str] = field(default_factory=list)   # index keys used in this cell


@dataclass
class Knob:
    name: str
    value: str
    comment: str
    cost: bool     # comment carries the compute-cost marker "↓"
    derived: bool  # computed from other config names and uncommented (e.g. P, RUN_NAME) → hidden in the page (Q2 of the questions after Step 1)


@dataclass
class Notebook:
    exp: str
    path: str
    intro_md: str
    steps: List[Step]
    knobs: List[Knob]
    imported_unused: List[str]


def load_notebook(exp: str, rel: str) -> Notebook:
    nb = json.loads((REPO / rel).read_text(encoding="utf-8"))
    intro, steps = "", []
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            if steps:
                raise ValueError(f"{rel}: markdown cell after code cells is not supported")
            intro += src
            continue
        first = src.splitlines()[0] if src.strip() else ""
        m = CELL_HEADER.match(first)
        if not m:
            raise ValueError(f"{rel}: code cell without '# ── Cell N · Title' header: {first[:60]!r}")
        steps.append(Step(int(m.group(1)), m.group(2), src))
    nums = [s.number for s in steps]
    if nums != list(range(1, len(steps) + 1)):
        raise ValueError(f"{rel}: cell numbers {nums} are not 1..{len(steps)}")
    config = next((s for s in steps if s.title.startswith("Config")), None)
    knobs = parse_config(config.code) if config else []
    return Notebook(exp, rel, intro, steps, knobs, [])


def parse_config(code: str) -> List[Knob]:
    """Top-level `NAME = value  # comment` lines (UPPER names) with their continuation comment lines."""
    tree = ast.parse(code)
    lines = code.splitlines()
    comments: Dict[int, str] = {}   # line → comment text (without '#')
    for tok in tokenize.generate_tokens(io.StringIO(code).readline):
        if tok.type == tokenize.COMMENT:
            comments[tok.start[0]] = tok.string.lstrip("#").strip()
    knobs, seen = [], set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        tgt = node.targets[0]
        names = [tgt.id] if isinstance(tgt, ast.Name) else \
                [e.id for e in tgt.elts if isinstance(e, ast.Name)] if isinstance(tgt, ast.Tuple) else []
        if not names or not all(n.isupper() for n in names):
            continue
        parts = [comments[i] for i in range(node.lineno, node.end_lineno + 1) if i in comments]
        i = node.end_lineno + 1   # continuation: following comment-only lines that are indented
        while i <= len(lines) and lines[i - 1].startswith(" ") and lines[i - 1].strip().startswith("#"):
            parts.append(comments.get(i, ""))
            i += 1
        comment = " ".join(p for p in parts if p)
        value = ast.get_source_segment(code, node.value) or ""
        refs = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)} & seen
        knobs.append(Knob(", ".join(names), value, comment, "↓" in comment, bool(refs) and not comment))
        seen.update(names)
    return knobs


def find_helpers(nb: Notebook, index: Dict[str, Helper]) -> None:
    """Fill Step.helpers: module attributes (metrics.x), imported names (set_seed), Class.method on imported
    classes (UpsamplerLoader.load), and obj.method(...) when the method name is unique to one anyup class."""
    method_owner: Dict[str, List[str]] = {}
    for key, h in index.items():
        if h.kind == "method":
            method_owner.setdefault(key.split(".")[1], []).append(key)
    imported_mods: Dict[str, str] = {}    # alias → short module name
    imported_names: Dict[str, str] = {}   # local name → index key
    for step in nb.steps:
        for node in ast.walk(ast.parse(step.code)):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("anyup"):
                for a in node.names:
                    full = f"{node.module}.{a.name}"
                    if any(full == m for m, _ in HELPER_FILES):
                        imported_mods[a.asname or a.name] = a.name
                    else:
                        imported_names[a.asname or a.name] = f"{node.module.split('.')[-1]}.{a.name}"
    used_any = set()
    for step in nb.steps:
        found: List[str] = []
        for node in ast.walk(ast.parse(step.code)):
            key = None
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                base = node.value.id
                if base in imported_mods:
                    key = f"{imported_mods[base]}.{node.attr}"
                elif base in imported_names and f"{base}.{node.attr}" in index:
                    key = f"{base}.{node.attr}"
                    used_any.add(base)
            if key is None and isinstance(node, ast.Attribute) and node.attr not in GENERIC_METHODS \
                    and len(method_owner.get(node.attr, [])) == 1 and not node.attr.startswith("_"):
                key = method_owner[node.attr][0]
            if key is None and isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) \
                    and node.id in imported_names:
                key = imported_names[node.id]
                used_any.add(node.id)
            if key is not None:
                if key not in index:
                    raise KeyError(f"{nb.path} cell {step.number}: {key} not found in the anyup source")
                if key not in found:
                    found.append(key)
        # "UpsamplerLoader.load" already names the class → drop the bare "upsamplers.UpsamplerLoader"
        classes_with_method = {k.split(".")[0] for k in found if index[k].kind == "method"}
        step.helpers = [k for k in found if not (index[k].kind == "class" and k.split(".")[1] in classes_with_method)]
    nb.imported_unused = sorted(n for n in imported_names if n not in used_any)


# ── Git ──────────────────────────────────────────────────────────────────────
def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def git_state() -> Dict:
    dirty = git("status", "--porcelain", "--", "anyup", "experiments").splitlines()
    return dict(commit=git("rev-parse", "HEAD"), short=git("rev-parse", "--short", "HEAD"),
                branch=git("rev-parse", "--abbrev-ref", "HEAD"), dirty=dirty)


# ── Report (Step 1) ──────────────────────────────────────────────────────────
def collect():
    index = build_index()
    nbs = []
    for exp, rel in NOTEBOOKS.items():
        nb = load_notebook(exp, rel)
        find_helpers(nb, index)
        for step in nb.steps:
            for key in step.helpers:
                index[key].used_by.setdefault(exp, []).append(step.number)
        nbs.append(nb)
    return index, nbs


def report(index: Dict[str, Helper], nbs: List[Notebook], state: Dict) -> None:
    print(f"commit {state['short']} on {state['branch']}; dirty files in anyup/ experiments/: {len(state['dirty'])}")
    for line in state["dirty"]:
        print("   ", line)
    print(f"index: {len(index)} definitions in {len(HELPER_FILES)} files")
    for nb in nbs:
        print(f"\n== {nb.exp}  {nb.path}: {len(nb.steps)} code cells, {len(nb.knobs)} config knobs "
              f"({sum(k.cost for k in nb.knobs)} with ↓), intro {len(nb.intro_md.splitlines())} lines")
        for s in nb.steps:
            print(f"  Cell {s.number} · {s.title}  [{len(s.code.splitlines())} lines]")
            if s.helpers:
                print("      " + ", ".join(s.helpers))
        if nb.imported_unused:
            print(f"  imported but not used: {', '.join(nb.imported_unused)}")
        print("  knobs:", ", ".join(("↓" if k.cost else "") + k.name for k in nb.knobs if not k.derived))
        print("  hidden (derived):", ", ".join(k.name for k in nb.knobs if k.derived))
    used = [h for h in index.values() if h.used_by]
    print(f"\nshared helpers used: {len(used)}")
    for h in sorted(used, key=lambda h: (h.file, h.lineno)):
        print(f"  {h.qualname:<36} {h.kind:<8} {h.file}:{h.lineno}-{h.end_lineno}  "
              f"used by {', '.join(f'{e}{c}' for e, c in sorted(h.used_by.items()))}")


# ── HTML (Steps 2–4) ─────────────────────────────────────────────────────────
def esc(x) -> str:
    return html.escape(str(x), quote=True)


def inline(text: str) -> str:
    """Escape, then `code` → <code>."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", esc(text))


def gh_link(state: Dict, path: str, a: Optional[int] = None, b: Optional[int] = None) -> str:
    anchor = "" if a is None else f"#L{a}" if b in (None, a) else f"#L{a}-L{b}"
    return f"{GITHUB}/blob/{state['commit']}/{path}{anchor}"


def helper_id(key: str) -> str:
    return "h-" + key.replace(".", "-")


def step_id(exp: str, n: int) -> str:
    return f"{exp}-cell{n}"


class Highlighter:
    def __init__(self):
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import PythonLexer
        from pygments import highlight
        self._hl, self._lexer = highlight, PythonLexer()
        self._fmt = HtmlFormatter(cssclass="hl", nowrap=False)
        light = HtmlFormatter(style="default").get_style_defs(".hl")
        dark = HtmlFormatter(style="github-dark").get_style_defs(":root[data-theme=dark] .hl")
        # pygments sets its own block background; the page's token --code-bg is used instead
        strip = lambda css: "\n".join(l for l in css.splitlines() if not re.match(r".*\.hl\s*\{\s*background", l)
                                      and "pre { line-height" not in l)
        self.css = strip(light) + "\n" + strip(dark)

    def __call__(self, code: str) -> str:
        return self._hl(code, self._lexer, self._fmt)


def fmt_num(v, nd=4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return esc(v)


def fmt_p(p) -> str:
    return "—" if p is None else f"{p:.2e}"


def table(columns: List[str], rows: List[List[str]], cls: str = "") -> str:
    """Cells are already-escaped HTML."""
    head = "".join(f"<th>{c}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def load_results(names: List[str]) -> Dict[str, object]:
    out = {}
    for n in names:
        p = RESULTS_DIR / n
        if not p.exists():
            continue
        if n.endswith(".json"):
            out[n] = json.loads(p.read_text(encoding="utf-8"))
        elif n.endswith(".csv"):
            with open(p, newline="", encoding="utf-8") as f:
                out[n] = list(csv.DictReader(f))
    return out


def paired_table(paired: Dict[str, Dict], label: str = "AnyUp3D vs") -> str:
    rows = []
    for base, st in paired.items():
        rows.append([f"{esc(label)} {esc(base)}", f"{st['mean_delta']:+.4f}", f"{st['n_improved']} / {st['n']}",
                     fmt_p(st.get("sign_p_two_sided")), fmt_p(st.get("wilcoxon_p_two_sided"))])
    return table(["Comparison", "Mean Δ", "Improved", "Sign p (2-sided)", "Wilcoxon p (2-sided)"], rows, "num")


def provenance_line(js: Dict) -> str:
    env, cfg = js.get("env", {}), js.get("config", {})
    run = cfg.get("RUN_NAME") or cfg.get("run_name") or "—"
    commit = (env.get("git_commit") or "")[:7] or "—"
    dirty = ' <span class="warn">uncommitted changes at run time</span>' if env.get("git_dirty") else ""
    return (f'<p class="meta">Run <code>{esc(run)}</code> · commit <code>{esc(commit)}</code>{dirty} · '
            f'{esc(env.get("gpu") or "—")} · {esc(env.get("time_utc") or "—")}</p>')


def render_results(exp: str, res: Dict[str, object]) -> str:
    js = res.get(f"{exp}_summary.json")
    if js is None:
        return ""
    s = js["summary"]
    parts = [provenance_line(js)]
    if exp == "exp0":
        rows = [[esc(n), f"{v['params']:,}", f"{v['size_mib']:.2f}"] for n, v in s["params"].items()]
        parts.append(table(["Upsampler", "Parameters", "Size (MiB, fp32)"], rows, "num"))
        rows = [[esc(m), f"{r['params_increase_pct']:+.1f} %", f"{r['time_ratio']:.2f}×",
                 f"{r['memory_ratio_extra']:.2f}×", f"{r['memory_ratio_total']:.2f}×"]
                for m, r in s["ratios_vs_anyup2d"].items()]
        parts.append("<h4>AnyUp3D / AnyUp2D, main row</h4>" +
                     table(["Mode", "Parameters", "Time", "Extra peak memory", "Total peak memory"], rows, "num"))
        csv_rows = res.get("exp0_per_item.csv")
        if csv_rows:
            main = [r for r in csv_rows if r["part"] == "main"]
            rows = [[esc(r["mode"]), esc(r["upsampler"]),
                     *( [f"{float(r['median_ms']):.2f}", f"{float(r['iqr_ms']):.2f}", f"{float(r['per_frame_ms']):.2f}",
                         f"{float(r['peak_total_mib']):.1f}", f"{float(r['peak_extra_mib']):.1f}"]
                        if r["status"] == "ok" else ["OOM"] + ["—"] * 4)] for r in main]
            parts.append("<h4>Main row</h4>" + table(
                ["Mode", "Upsampler", "Median ms", "IQR ms", "ms / frame", "Peak MiB", "Extra MiB"], rows, "num"))
        n_fail = sum(not c["passed"] for c in s.get("checks", []))
        parts.append(f'<p class="meta">{len(s.get("checks", []))} checks, {n_fail} failed.</p>')
    elif exp == "exp1":
        rows = [[esc(c), " / ".join(f"{v:.2f}" for v in d["miou_per_seed"].values()),
                 f"{d['miou_mean']:.2f} ± {d['miou_std']:.2f}", f"{d['image_miou_mean']:.2f}"]
                for c, d in s["conditions"].items()]
        parts.append(table(["Condition", "mIoU per seed", "mIoU mean ± std", "Mean per-image mIoU"], rows, "num"))
        if s.get("paired_images"):
            parts.append(f"<h4>Paired, per image ({s.get('n_images_scored', '—')} images)</h4>" +
                         paired_table({k.replace("anyup3d_vs_", ""): v for k, v in s["paired_images"].items()}))
    elif exp == "exp2":
        rows = []
        for c, d in s["per_condition"].items():
            cell = lambda k: "—" if d[k]["mean"] is None else f"{d[k]['mean']:.4f} ± {d[k]['std']:.4f}"
            rows.append([esc(c), cell("all"), cell("within"), cell("cross")])
        parts.append(table(["Condition", "All pairs", "Within chunk", "Across chunks"], rows, "num"))
        if s.get("paired_anyup3d_vs"):
            parts.append("<h4>Paired, per clip (all pairs)</h4>" + paired_table(s["paired_anyup3d_vs"]))
    elif exp == "exp3":
        keys = ["J&F-Mean", "J-Mean", "J-Recall", "J-Decay", "F-Mean", "F-Recall", "F-Decay"]
        rows = [[esc(c)] + [f"{d[k]:.4f}" for k in keys] + [str(d["n_objects"])] for c, d in s["per_condition"].items()]
        parts.append(table(["Condition"] + keys + ["Objects"], rows, "num"))
        if s.get("paired_anyup3d_vs"):
            parts.append("<h4>Paired, per clip (J&amp;F)</h4>" + paired_table(s["paired_anyup3d_vs"]))
    return "\n".join(parts)


def render_flow(flow) -> str:
    items = "".join(f'<li><span class="stage">{inline(a)}</span><span class="shape">{inline(b)}</span></li>' for a, b in flow)
    return f'<ol class="flow">{items}</ol>'


def render_chips(keys: List[str], implicit: bool) -> str:
    cls = "chip implicit" if implicit else "chip"
    title = ' title="called indirectly (through an object)"' if implicit else ""
    return "".join(f'<a class="{cls}" href="#{helper_id(k)}"{title}>{esc(k)}</a>' for k in keys)


def render_experiment(nb: "Notebook", c: Dict, index: Dict[str, Helper], state: Dict, hl: Highlighter,
                      res: Dict[str, object]) -> str:
    exp = nb.exp
    out = [f'<section id="{exp}" class="exp">',
           f'<h2>{esc(c["title"])}</h2>',
           f'<p class="meta"><a href="{gh_link(state, nb.path)}">{esc(nb.path)}</a> · {esc(c["provenance"])}</p>']
    out += [f"<p>{inline(p)}</p>" for p in c["purpose"]]
    out.append(f'<h3 id="{exp}-flow">Data flow</h3>' + render_flow(c["flow"]) + f'<p class="note">{inline(c["flow_note"])}</p>')

    knobs = [k for k in nb.knobs if not k.derived]
    rows = [[f'<code>{esc(k.name)}</code>' + (' <span class="cost" title="compute-cost setting">↓</span>' if k.cost else ""),
             f'<code class="val">{esc(k.value)}</code>', inline(k.comment) if k.comment else ""] for k in knobs]
    out.append(f'<h3 id="{exp}-config">Configuration</h3>'
               f'<p class="note">Settings of the config cell (Cell 2) at the documented commit. '
               f'<span class="cost">↓</span> marks a compute-cost setting; its comment names the settings that depend on it. '
               + (f'{n_derived} derived names (run names and paths) are not listed.' if (n_derived := sum(k.derived for k in nb.knobs)) else "")
               + '</p>'
               + table(["Name", "Value", "Comment"], rows, "config"))

    out.append(f'<h3 id="{exp}-steps">Steps</h3>')
    for st in nb.steps:
        desc = c["steps"].get(st.number)
        if desc is None:
            raise KeyError(f"doc_content: {exp} has no description for Cell {st.number}")
        imp = c.get("implicit", {}).get(st.number, [])
        for k in imp:
            if k not in index:
                raise KeyError(f"doc_content: {exp} Cell {st.number} implicit helper {k} not in the index")
        chips = render_chips(st.helpers, False) + render_chips([k for k in imp if k not in st.helpers], True)
        n_lines = len(st.code.splitlines())
        out.append(
            f'<article class="step" id="{step_id(exp, st.number)}">'
            f'<h4><span class="num">{st.number}</span>{esc(st.title)}</h4>'
            f'<p>{inline(desc)}</p>'
            + (f'<div class="chips">{chips}</div>' if chips else "")
            + f'<details class="code"><summary>Code · {n_lines} lines</summary>'
              f'<div class="codewrap"><button class="copy" type="button">Copy</button>{hl(st.code)}</div></details>'
              f'</article>')

    out.append(f'<h3 id="{exp}-results">Results</h3>')
    rendered = render_results(exp, res)
    if rendered:
        out.append(rendered)
    else:
        files = ", ".join(f"<code>{esc(f)}</code>" for f in c["results_files"])
        out.append(f'<p class="pending">Pending. Copy {files} from Drive <code>{esc(c["drive_dir"])}</code> '
                   f'into <code>docs/experiments/results/</code> and rebuild.</p>')
    out.append("<h4>Thesis reference</h4>" + table(["Quantity", "Thesis"], [[esc(a), esc(b)] for a, b in c["thesis"]]))
    out.append('<h3>Protocol vs thesis</h3><ul class="plain">' + "".join(f"<li>{inline(p)}</li>" for p in c["protocol"]) + "</ul>")
    out.append('<h3>Known issues</h3><ul class="issues">' +
               "".join(f'<li><span class="tag">{esc(t)}</span>{inline(x)}</li>' for t, x in c["issues"]) + "</ul>")
    out.append("</section>")
    return "\n".join(out)


def render_helpers(index: Dict[str, Helper], used_keys: List[str], state: Dict, hl: Highlighter,
                   titles: Dict[str, str]) -> str:
    by_file: Dict[str, List[Helper]] = {}
    for k in used_keys:
        by_file.setdefault(index[k].file, []).append(index[k])
    out = ['<section id="shared" class="exp"><h2>Shared modules</h2>',
           '<p>Every function, class and constant of the <code>anyup</code> package that the notebooks call, '
           'directly or through an object (dashed chips in the steps). Links point to the source at the documented commit.</p>']
    order = [f for _, f in HELPER_FILES]
    for f in sorted(by_file, key=order.index):
        out.append(f'<h3 id="file-{esc(f.replace("/", "-"))}"><a href="{gh_link(state, f)}">{esc(f)}</a></h3>')
        for h in sorted(by_file[f], key=lambda h: h.lineno):
            key = next(k for k, v in index.items() if v is h)
            used = " ".join(f'<a href="#{step_id(e, n)}">{esc(titles[e].split(" · ")[0])} · Cell {n}</a>'
                            for e, cells in sorted(h.used_by.items()) for n in cells)
            doc = f'<pre class="doc">{esc(h.doc)}</pre>' if h.doc else ""
            out.append(
                f'<article class="helper" id="{helper_id(key)}">'
                f'<h4><code>{esc(key)}</code><span class="kind">{esc(h.kind)}</span></h4>'
                f'<p class="meta"><a href="{gh_link(state, h.file, h.lineno, h.end_lineno)}">{esc(h.file)}:{h.lineno}–{h.end_lineno}</a>'
                f' · used in {used}</p>'
                f'{hl(h.signature)}{doc}</article>')
    out.append("</section>")
    return "\n".join(out)


CSS = r"""
:root{--bg:#fbfbfa;--surface:#ffffff;--text:#1d1f22;--muted:#5d636b;--line:#e3e4e2;--accent:#2a5bd7;
--accent-soft:#e8eefc;--code-bg:#f5f5f3;--warn:#a35200;--warn-soft:#fdf0e3;--ok:#1d7a46;
--link:#1f56d6;--link-line:rgba(31,86,214,.35);--chip-bg:#eef3fd;--chip-line:#b9cbf3;--divider:#c3c7cd;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
--sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#15171a;--surface:#1c1f23;--text:#e4e6e8;
--muted:#9aa1a9;--line:#2d3137;--accent:#7aa2ff;--accent-soft:#1f2a44;--code-bg:#111316;--warn:#f0a35c;--warn-soft:#3a2a1a;--ok:#5cc58a;
--link:#8cb0ff;--link-line:rgba(140,176,255,.4);--chip-bg:#1b2438;--chip-line:#34477a;--divider:#454b54}}
:root[data-theme=dark]{--bg:#15171a;--surface:#1c1f23;--text:#e4e6e8;--muted:#9aa1a9;--line:#2d3137;--accent:#7aa2ff;
--accent-soft:#1f2a44;--code-bg:#111316;--warn:#f0a35c;--warn-soft:#3a2a1a;--ok:#5cc58a;
--link:#8cb0ff;--link-line:rgba(140,176,255,.4);--chip-bg:#1b2438;--chip-line:#34477a;--divider:#454b54}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:16px}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 var(--sans)}
a{color:var(--link);text-decoration:none}a:hover{text-decoration:underline}
main a:not(.chip){text-decoration:underline;text-decoration-color:var(--link-line);text-underline-offset:3px}
main a:not(.chip):hover{text-decoration-color:currentColor}
main a code{color:inherit}
code{font:13px var(--mono);background:var(--code-bg);padding:1px 4px;border-radius:4px;overflow-wrap:anywhere}
.layout{display:grid;grid-template-columns:250px minmax(0,1fr);min-height:100vh}
nav.side{position:sticky;top:0;height:100vh;overflow:auto;border-right:1px solid var(--line);padding:60px 18px 24px;background:var(--surface)}
button.navtoggle{position:fixed;top:12px;left:12px;z-index:10;display:inline-flex;align-items:center;gap:6px;font:13px var(--sans);color:var(--text);background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:5px 10px;cursor:pointer}
button.navtoggle:hover{border-color:var(--link);color:var(--link)}
body.nav-closed .layout{grid-template-columns:minmax(0,1fr)}
body.nav-closed nav.side{display:none}
body.nav-closed main{margin:0 auto}
nav.side .brand{font-weight:650;font-size:15px;margin-bottom:2px}
nav.side .sub{color:var(--muted);font-size:12px;margin-bottom:18px}
nav.side ul{list-style:none;margin:0;padding:0}
nav.side li a{display:block;padding:5px 8px;border-radius:6px;color:var(--text);font-size:14px}
nav.side li a:hover{background:var(--accent-soft);text-decoration:none}
nav.side li a.active{background:var(--accent-soft);color:var(--accent);font-weight:600}
nav.side li.subitem a{padding-left:20px;font-size:13px;color:var(--muted)}
nav.side .row{display:flex;align-items:center;gap:2px}
nav.side .row a{flex:1;min-width:0}
nav.side button.caret{flex:none;width:26px;height:26px;border:none;background:none;color:var(--muted);border-radius:6px;cursor:pointer;font-size:12px;transition:transform .15s}
nav.side button.caret:hover{background:var(--accent-soft);color:var(--link)}
nav.side button.caret[aria-expanded=true]{transform:rotate(90deg)}
nav.side li.grp:not(.open)>ul.subs{display:none}
nav.side .expall{display:flex;gap:6px;margin:0 0 10px}
nav.side .expall button{font:12px var(--sans);color:var(--muted);background:none;border:1px solid var(--line);border-radius:6px;padding:2px 8px;cursor:pointer}
nav.side .expall button:hover{color:var(--link);border-color:var(--link)}
nav.side .theme{margin-top:20px;font:inherit;font-size:12px;color:var(--muted);background:none;border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
main{padding:40px 48px 80px;max-width:980px}
h1{font-size:28px;line-height:1.25;margin:0 0 6px;font-weight:680;letter-spacing:-.01em}
h2{font-size:22px;margin:0 0 4px;font-weight:650;letter-spacing:-.005em}
h3{font-size:16px;margin:30px 0 10px;font-weight:620}
h4{font-size:14px;margin:18px 0 8px;font-weight:620}
p{margin:0 0 12px;max-width:72ch}
.lead{color:var(--muted);font-size:16px;max-width:70ch}
.meta{color:var(--muted);font-size:13px}
.note{color:var(--muted);font-size:13px;max-width:80ch}
section.exp{padding-top:40px;margin-top:72px;border-top:6px solid var(--divider)}
section.exp>h2{font-size:26px}
.banner{background:var(--warn-soft);color:var(--warn);border-radius:8px;padding:10px 14px;margin:16px 0;font-size:14px}
.tablewrap{overflow-x:auto;margin:8px 0 16px;border:1px solid var(--line);border-radius:8px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:left;padding:7px 12px;border-bottom:1px solid var(--line);vertical-align:top}
tbody tr:last-child td{border-bottom:none}
th{font-weight:600;color:var(--muted);font-size:12.5px;background:var(--code-bg)}
table.num td:not(:first-child){font-variant-numeric:tabular-nums;white-space:nowrap}
table.config td:first-child{white-space:nowrap}
table.config td:nth-child(2){min-width:230px;max-width:340px}
table.config code.val{white-space:pre-wrap;overflow-wrap:anywhere}
table.overview td:first-child{white-space:nowrap}
table.config td:nth-child(3){color:var(--muted);font-size:13px}
code.val{background:none;padding:0}
.cost{color:var(--warn);font-weight:700}
ol.flow{list-style:none;padding:0;margin:8px 0 10px;display:flex;flex-wrap:wrap;gap:8px 0;align-items:stretch}
ol.flow li{display:flex;flex-direction:column;justify-content:center;background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:7px 12px;position:relative;margin-right:26px;min-width:0}
ol.flow li:not(:last-child)::after{content:"→";position:absolute;right:-20px;top:50%;transform:translateY(-50%);color:var(--muted)}
ol.flow .stage{font-weight:600;font-size:13px}
ol.flow .shape{font:12px var(--mono);color:var(--muted);overflow-wrap:anywhere}
ol.flow .shape code,ol.flow .stage code{background:none;padding:0}
article.step{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 18px 12px;margin:0 0 12px}
article.step h4{margin:0 0 6px;display:flex;align-items:baseline;gap:10px}
article.step .num{display:inline-flex;align-items:center;justify-content:center;min-width:22px;height:22px;border-radius:11px;background:var(--accent-soft);color:var(--accent);font-size:12px;font-weight:700}
article.step p{font-size:14px;margin-bottom:8px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 8px}
.chip{font:12px var(--mono);border:1px solid var(--chip-line);background:var(--chip-bg);border-radius:999px;padding:2px 9px;color:var(--link)}
.chip:hover{border-color:var(--link);text-decoration:none}
.chip.implicit{border-style:dashed;background:transparent;color:var(--link)}
details.code summary{cursor:pointer;color:var(--muted);font-size:13px;user-select:none;padding:2px 0}
details.code summary:hover{color:var(--accent)}
.codewrap{position:relative;margin-top:8px}
.hl{background:var(--code-bg);border-radius:8px;overflow-x:auto;margin:0}
.hl pre{margin:0;padding:12px 14px;font:12.5px/1.5 var(--mono);background:none}
button.copy{position:absolute;top:8px;right:8px;font:12px var(--sans);color:var(--muted);background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:2px 9px;cursor:pointer;z-index:1}
button.copy:hover{color:var(--accent);border-color:var(--accent)}
.pending{background:var(--code-bg);border:1px dashed var(--line);border-radius:8px;padding:10px 14px;font-size:14px;max-width:none}
.warn{color:var(--warn)}
ul.plain,ul.issues{padding-left:20px;margin:0 0 12px}
ul.plain li,ul.issues li{margin-bottom:6px;font-size:14px;max-width:80ch}
ul.issues{list-style:none;padding-left:0}
ul.issues .tag{display:inline-block;font:600 11.5px var(--mono);color:var(--warn);background:var(--warn-soft);border-radius:4px;padding:1px 6px;margin-right:8px}
article.helper{border-top:1px solid var(--line);padding:14px 0 6px}
article.helper h4{margin:0 0 2px;display:flex;gap:10px;align-items:baseline}
article.helper h4 code{background:none;padding:0;font-size:14px;font-weight:600}
article.helper .kind{font-size:11.5px;color:var(--muted);border:1px solid var(--line);border-radius:4px;padding:0 6px}
article.helper .meta a+a{margin-left:4px}
pre.doc{white-space:pre-wrap;font:12.5px/1.5 var(--mono);color:var(--muted);margin:8px 0 0;padding:0 2px}
article.helper .hl pre{padding:8px 12px;white-space:pre-wrap}
:target{animation:flash 1.6s ease-out}
@keyframes flash{from{box-shadow:0 0 0 3px var(--accent-soft)}to{box-shadow:0 0 0 3px transparent}}
footer{margin-top:60px;color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:14px}
@media (max-width:860px){
 .layout{display:block}
 nav.side{position:static;height:auto;border-right:none;border-bottom:1px solid var(--line);padding:56px 16px 14px}
 nav.side .sub{margin-bottom:8px}
 main{padding:60px 16px 60px}
 h1{font-size:23px}
}
"""

JS = r"""
(function(){
  var root=document.documentElement, KEY="anyup-docs-theme";
  function get(){try{return localStorage.getItem(KEY)}catch(e){return null}}
  function set(v){try{localStorage.setItem(KEY,v)}catch(e){}}
  var saved=get(); if(saved){root.setAttribute("data-theme",saved);}
  var NAVKEY="anyup-docs-nav", nb=document.getElementById("navtoggle");
  function navGet(){try{return localStorage.getItem(NAVKEY)}catch(e){return null}}
  function navSet(v){try{localStorage.setItem(NAVKEY,v)}catch(e){}}
  function setNav(open){document.body.classList.toggle("nav-closed",!open);nb.setAttribute("aria-expanded",open?"true":"false");
    nb.textContent=open?"\u2715 Hide menu":"\u2630 Menu";}
  var ns=navGet(); setNav(ns? ns==="open" : !matchMedia("(max-width:860px)").matches);
  nb.addEventListener("click",function(){var o=document.body.classList.contains("nav-closed");setNav(o);navSet(o?"open":"closed");});
  var btn=document.getElementById("theme");
  function label(){var d=root.getAttribute("data-theme")||(matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light");
    btn.textContent=d==="dark"?"Light theme":"Dark theme";return d;}
  label();
  btn.addEventListener("click",function(){var n=label()==="dark"?"light":"dark";root.setAttribute("data-theme",n);set(n);label();});
  document.querySelectorAll("button.copy").forEach(function(b){
    b.addEventListener("click",function(){
      var pre=b.parentNode.querySelector("pre"); var t=pre?pre.innerText:"";
      var done=function(){b.textContent="Copied";setTimeout(function(){b.textContent="Copy"},1200)};
      if(navigator.clipboard){navigator.clipboard.writeText(t).then(done,function(){})}
    });
  });
  var GKEY="anyup-docs-open-groups", groups=[].slice.call(document.querySelectorAll("nav.side li.grp"));
  function gGet(){try{return JSON.parse(localStorage.getItem(GKEY)||"[]")}catch(e){return []}}
  function gSave(){try{localStorage.setItem(GKEY,JSON.stringify(groups.filter(function(g){return g.classList.contains("open")}).map(function(g){return g.dataset.exp})))}catch(e){}}
  function gSet(g,open){g.classList.toggle("open",open);var b=g.querySelector("button.caret");b.setAttribute("aria-expanded",open?"true":"false");b.setAttribute("aria-label",open?"Hide topics":"Show topics");}
  var openSaved=gGet(); groups.forEach(function(g){gSet(g,openSaved.indexOf(g.dataset.exp)>=0);
    g.querySelector("button.caret").addEventListener("click",function(){gSet(g,!g.classList.contains("open"));gSave();});});
  document.getElementById("expandall").addEventListener("click",function(){groups.forEach(function(g){gSet(g,true)});gSave();});
  document.getElementById("collapseall").addEventListener("click",function(){groups.forEach(function(g){gSet(g,false)});gSave();});
  var links=[].slice.call(document.querySelectorAll("nav.side li:not(.subitem) > a, nav.side .row > a"));
  var secs=links.map(function(a){return document.querySelector(a.getAttribute("href"))}).filter(Boolean);
  if("IntersectionObserver" in window){
    var obs=new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting){
      links.forEach(function(a){a.classList.toggle("active",a.getAttribute("href")==="#"+e.target.id)})}})},
      {rootMargin:"-10% 0px -80% 0px"});
    secs.forEach(function(s){obs.observe(s)});
  }
  if(location.hash){var t=document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if(t){var d=t.closest("details");if(d)d.open=true;}}
})();
"""


def render_page(index: Dict[str, Helper], nbs: List["Notebook"], state: Dict, dirty_banner: bool) -> str:
    import doc_content as C
    hl = Highlighter()
    titles = {e: C.EXPERIMENTS[e]["title"] for e in NOTEBOOKS}
    res = load_results(sorted({f for e in NOTEBOOKS for f in C.EXPERIMENTS[e]["results_files"]}))
    used_keys = sorted({k for nb in nbs for s in nb.steps for k in s.helpers} |
                       {k for e in NOTEBOOKS for ks in C.EXPERIMENTS[e].get("implicit", {}).values() for k in ks})
    # implicit helpers count as used by their step (for the "used in" links)
    for e in NOTEBOOKS:
        for n, ks in C.EXPERIMENTS[e].get("implicit", {}).items():
            for k in ks:
                if n not in index[k].used_by.setdefault(e, []):
                    index[k].used_by[e].append(n)
                    index[k].used_by[e].sort()

    nav = ['<li><a href="#overview">Overview</a></li>']
    for e in NOTEBOOKS:
        subs = "".join(f'<li class="subitem"><a href="#{e}-{sub}">{lab}</a></li>'
                       for sub, lab in (("flow", "Data flow"), ("config", "Configuration"), ("steps", "Steps"), ("results", "Results")))
        nav.append(f'<li class="grp" data-exp="{e}"><div class="row"><a href="#{e}">{esc(C.EXPERIMENTS[e]["nav"])}</a>'
                   f'<button class="caret" type="button" aria-expanded="false" aria-controls="sub-{e}" '
                   f'aria-label="Show topics">&#x25B8;</button></div><ul class="subs" id="sub-{e}">{subs}</ul></li>')
    nav.append('<li><a href="#shared">Shared modules</a></li>')

    ov = C.OVERVIEW_TABLE
    ov_rows = [[f'<a href="#{r[0]}">{esc(C.EXPERIMENTS[r[0]]["nav"].split(" · ")[0])}</a>'] + [esc(x) for x in r[1:]]
               for r in ov["rows"]]
    banner = ('<div class="banner">Built with uncommitted changes in <code>anyup/</code> or <code>experiments/</code>: '
              'code on this page may not match the linked source.</div>' if dirty_banner else "")
    body = [
        f'<header id="overview"><h1>{esc(C.TITLE)}</h1><p class="lead">{esc(C.SUBTITLE)}</p>',
        f'<p class="meta">Documented commit <a href="{GITHUB}/tree/{state["commit"]}"><code>{state["short"]}</code></a> '
        f'on <code>{esc(state["branch"])}</code></p>{banner}</header>',
        *[f"<p>{inline(p)}</p>" for p in C.OVERVIEW],
        table(ov["columns"], ov_rows, "overview"),
        "<h3>Reproducibility</h3>", *[f"<p>{inline(p)}</p>" for p in C.REPRODUCIBILITY],
    ]
    for nb in nbs:
        body.append(render_experiment(nb, C.EXPERIMENTS[nb.exp], index, state, hl, res))
    body.append(render_helpers(index, used_keys, state, hl, titles))
    body.append(f'<footer>Generated by <code>docs/experiments/build_docs.py</code> from the notebooks and source at commit '
                f'<code>{state["short"]}</code>. Do not edit this file by hand; edit <code>doc_content.py</code> or the code and rebuild.</footer>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AnyUp3D Experiments</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 16 16%27%3E%3Crect width=%2716%27 height=%2716%27 rx=%273%27 fill=%27%232a5bd7%27/%3E%3C/svg%3E">
<style>{CSS}
{hl.css}</style></head>
<body><button class="navtoggle" id="navtoggle" type="button" aria-controls="sidenav" aria-expanded="true">&#x2715; Hide menu</button><div class="layout">
<nav class="side" id="sidenav"><div class="brand">AnyUp3D experiments</div><div class="sub">Exp 0–3 · commit {state["short"]}</div>
<div class="expall"><button type="button" id="expandall">Expand all</button><button type="button" id="collapseall">Collapse all</button></div><ul>{"".join(nav)}</ul><button class="theme" id="theme" type="button">Dark theme</button></nav>
<main>
{chr(10).join(body)}
</main></div>
<script>{JS}</script>
</body></html>
"""


# ── Build checks ─────────────────────────────────────────────────────────────
def build_checks(page: str, index: Dict[str, Helper], nbs: List["Notebook"], state: Dict) -> List[tuple]:
    checks = []
    # 1. every code cell rendered exactly once
    for nb in nbs:
        n_rendered = sum(page.count(f'id="{step_id(nb.exp, s.number)}"') for s in nb.steps)
        checks.append((f"{nb.exp}: every code cell once", n_rendered == len(nb.steps), f"{n_rendered}/{len(nb.steps)} cells"))
    # 2. every linked helper's def line exists at the pinned commit
    linked = [k for k in index if f'id="{helper_id(k)}"' in page]
    bad = []
    cache: Dict[str, List[str]] = {}
    for k in linked:
        h = index[k]
        if h.file not in cache:
            cache[h.file] = git("show", f"{state['commit']}:{h.file}").splitlines()
        name = k.split(".")[-1]
        line = cache[h.file][h.def_lineno - 1] if h.def_lineno <= len(cache[h.file]) else ""
        if not re.search(rf"\b(def|class)\s+{re.escape(name)}\b|^{re.escape(name)}\s*=", line.strip()):
            bad.append(f"{k}@{h.file}:{h.def_lineno}")
    checks.append(("helper links match the pinned commit", not bad, f"{len(linked) - len(bad)}/{len(linked)} ok" + (f"; bad: {bad}" if bad else "")))
    # 3. every chip points to a rendered helper card
    targets = set(re.findall(r'href="#(h-[^"]+)"', page))
    missing = sorted(t for t in targets if f'id="{t}"' not in page)
    checks.append(("every helper chip has a card", not missing, f"{len(targets)} targets" + (f"; missing {missing}" if missing else "")))
    return checks


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", action="store_true", help="print the parsed structure, write nothing")
    ap.add_argument("--allow-dirty", action="store_true", help="build although anyup/ or experiments/ have uncommitted changes")
    args = ap.parse_args(argv)
    state = git_state()
    index, nbs = collect()
    if args.report:
        report(index, nbs, state)
        return 0
    if state["dirty"] and not args.allow_dirty:
        print("anyup/ or experiments/ have uncommitted changes; links would point to different lines:")
        print("\n".join("   " + l for l in state["dirty"]))
        print("commit (and push) first, or pass --allow-dirty")
        return 1
    if not git("branch", "-r", "--contains", state["commit"]):
        print(f"WARNING: commit {state['short']} is not on any remote branch yet; GitHub links will 404 until it is pushed")
    sys.path.insert(0, str(DOCS_DIR))
    page = render_page(index, nbs, state, dirty_banner=bool(state["dirty"]))
    checks = build_checks(page, index, nbs, state)
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    results = sorted(p.name for p in RESULTS_DIR.glob("*")) if RESULTS_DIR.exists() else []
    print(f"results files found: {[r for r in results if not r.startswith('.')] or 'none (all Results sections show Pending)'}")
    if not all(ok for _, ok, _ in checks):
        print("checks failed; page NOT written")
        return 1
    OUT_HTML.write_text(page, encoding="utf-8")
    print(f"wrote {OUT_HTML.relative_to(REPO)} ({len(page) / 1024:.0f} KiB) for commit {state['short']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
