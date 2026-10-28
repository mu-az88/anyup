# Notebooks & Version Control

How to change notebooks such as `Train_3D_full.ipynb` without losing a working version.

## Rule: edit the same notebook, on a branch

Edit the existing notebook on a feature branch. Don't make a copy.

- **Git is the safety net.** The working version stays as it is on the base branch (e.g. `3Dconv`). If the edits break something, you can compare against that version or return to it, and the base branch is never affected.
- **The base branch only changes when you merge**, and you merge only after the changes are verified (see step 5).
- **Copies drift apart.** Duplicate notebooks (`train_3d_full (1).py`, `*_old.ipynb`, `*_FIXED.ipynb`) end up as diverging versions with no record of which one is right.

## Workflow

1. **Put the logic in `.py` modules and keep notebook edits small.**
   New logic goes into the `anyup/` package with pytest tests, and the notebook only imports and calls it. The tests run locally on CPU, so most bugs show up before Colab.

2. **Edit the notebook locally, then commit and push the branch.**

3. **Open the notebook in Colab straight from the branch**, with no copy on Drive:

   ```
   https://colab.research.google.com/github/mu-az88/anyup/blob/<branch>/Train_3D_full.ipynb
   ```

   Set `REPO_BRANCH` in the clone cell to the same branch. Otherwise the notebook is new but the `anyup/` code it imports is old.

4. **Run the debug pass** (the plan's "Done when" checks).
   If something fails, fix it locally, push again, and restart the Colab runtime so it clones fresh. The clone is skipped when `/content/anyup` already exists.

5. **Merge into the base branch only after everything passes.** Any follow-up branch starts from the updated base.

## Cautions

- **Don't make fixes inside Colab.** Changes made there live only in Colab, and you'd have to copy them back by hand. If you must edit there, save with *File → Save a copy in GitHub* to the same branch, and `git pull` before editing locally again.
- **Clear outputs before committing.** Saved outputs make diffs huge and hard to review, which is a problem when you're checking changes against `docs/DECISIONS.md`.
- **Log every change** in `docs/DECISIONS.md` (Date / Branch / Decision / Where / Affects).
