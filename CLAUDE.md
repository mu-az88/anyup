# How to Work With Me (Monzir)

I'm Monzir Hafez, an Electrical & Electronic Engineering (Software) student. My accepted M.Sc. thesis, AnyUp3D, is now being turned into a research paper.

## Communication

- Be direct, concise, and technical. No filler and no long preambles.
- Calibrate length to the question: short answers for quick questions, detailed ones for architecture and engineering problems.
- Flag errors clearly, including mine. Disagree when something is wrong.
- Give honest assessments. Never claim something works until it has been run and checked.
- When something is ambiguous, ask short multiple-choice questions before doing work. Don't guess on decisions that are mine to make, especially terminology and scope.
- Push back on unnecessary complexity. The simplest correct solution wins.

## Coding Rules

- **Targeted edits over rewrites.** Change only what's needed and show me what changed. Never regenerate a whole file or notebook unless I ask.
- **Verify before acting.** Check implementation-dependent facts (kernel sizes, resolutions, strides, shapes, checkpoint keys) against the actual code, not memory or assumptions.
- **Comment every compute-cost line.** Any line controlling batch size, resolution, number of frames, token length, chunk size, attention window, or kernel size gets a short comment, e.g. `# ↓ reduce to save memory`. The comment must also name the other lines that depend on this value, so I know what else to change to avoid crashes.
- **Regression-check refactors.** When refactoring experiment code, confirm the new path gives the same outputs as the old one on the same input before switching over.
- **Reproducibility.** Use fixed seeds, explicit configs, and results saved to disk (per-clip plus summary). No magic numbers buried in code.
- Keep shapes explicit in comments where tensors are reshaped or permuted, e.g. `# (T, C, h, w)`.
- Code must run on Google Colab (T4 / A100), with checkpoints on Google Drive.

## Working Style: Careful, Ask First

I work in a very careful style so the code doesn't break. I give tasks or commands; you **always ask me** before making any assumption or any decision that is not obvious. I will always answer. Never fill a gap silently.

The ideal flow for every task:

1. **Q&A round first.** Restate the task in one or two lines, read the relevant code, then ask me *everything* needed for the step (short multiple-choice questions where possible) before writing any code.
2. **Write the task file.** Record the task in `docs/plan/<plan>/TNN_<name>.md` (from `docs/plan/_templates/task_template.md`): task ID, goal, scope, every question with my answer, design, steps, tests, done-when. Add/update the task's row in the plan's Task Index. Show me the file and wait for my OK.
3. **Implement** in small steps, test each, and report results with actual numbers or shapes. Ideally no questions are needed here, but if anything unexpected or ambiguous comes up, **stop and ask** — don't guess. Log new answers in the task file.
4. **Close out.** Update the task's Implementation Log and status (in the task file and the plan's Task Index). Tell me what's left, what's uncertain, and what I need to verify.

**Keep the task file live during implementation.** The task file must always reflect what is *actually* happening, so that when the task is done it is an accurate report of what was really done, not just what was planned:

- **Changed decision** → update the original answer/design in place and note the change in the Implementation Log (old → new, and why).
- **Added or removed something** (files, functions, steps, tests, scope) → update the relevant section (Scope, Design, Steps, Tests, Done When) and log it.
- **After each implementation step** → add a log entry: what changed, files touched, test results with actual numbers/shapes.
- Do this as we go, in the same turn as the change, not in a batch at the end.

## Docs Structure

```
CLAUDE.md                          ← how we work (this file)
docs/
  context/
    project_context.md             ← always-loaded project facts and gotchas
  plan/
    _templates/task_template.md    ← copy this for every new task
    P1_experiments/                ← one folder per plan (P1, P2, ...)
      plan.md                      ← big picture for the plan + Task Index table
      T01_backbone_loader.md       ← one file per task (current step)
    P2_<name>/ ...                 ← next plan, e.g. training-pipeline fixes
```

- **Plans** are numbered `P1, P2, ...` in order of work: P1 experiments → P2 training-pipeline fixes → P3 scaling data and backbones.
- **Tasks** are numbered `T01, T02, ...` **globally** (never reused, P2 continues from the last P1 number). File name: `TNN_<snake_case_name>.md`.
- **Status** values: `todo` → `Q&A` → `in progress` → `done` (or `blocked`). Keep the task file and the plan's Task Index in sync.
- `docs/colab_training_guide.md` and `docs/training_guide.md` are older reference guides, not part of the plan.

## Writing Style (paper and reports)

- Use formal academic prose; paragraphs over bullet lists for discussion and conclusions.
- Keep it concise: slide or summary text is about 20–50 words unless I ask for more.
- The LaTeX bibliography style is IEEEtran (numeric).
- For LaTeX edits, give me the full current draft of the section or file after each edit, as a file.
- Frame limitations honestly but without apologizing. For example, low absolute J&F in the zero-shot setting is a lower bound, not a failure.
- Let me choose ambiguous terminology; offer options instead of deciding.

## Active Context

The files below are loaded automatically. **To switch context, edit these lines.** The project context stays; swap the plan and task lines as the work moves on.

@docs/context/project_context.md
@docs/plan/P1_experiments/plan.md
@docs/plan/P1_experiments/T01_backbone_loader.md
