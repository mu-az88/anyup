"""
tests/test_smoke_train3d.py

Subprocess smoke test for train_3d.py main().
Runs `python train_3d.py --config configs/anyup3d_train.yaml --debug`
and checks for a zero exit code.

Skipped automatically when required dependencies are missing:
  - anyup.modules.anyup3d (AnyUp3D)
  - anyup.modules.losses3d (combined_loss_3d)

Run:
    pytest tests/test_smoke_train3d.py -v
    pytest tests/test_smoke_train3d.py -v --run-slow   # if also gated behind slow
"""

import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent


_MISSING_DEPS = []
for _file in ["anyup/modules/anyup3d.py", "anyup/modules/losses3d.py"]:
    if not (_PROJECT_ROOT / _file).exists():
        _MISSING_DEPS.append(_file)

_skip_reason = (
    f"Missing dependencies: {', '.join(_MISSING_DEPS)}" if _MISSING_DEPS else ""
)


@pytest.mark.skipif(bool(_MISSING_DEPS), reason=_skip_reason)
def test_train3d_debug_smoke():
    """
    Run train_3d.py in --debug mode (10 steps, T=1, batch=1).
    Verifies the full training loop completes without error.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "train_3d.py"),
            "--config",
            str(_PROJECT_ROOT / "config" / "anyup3d_train.yaml"),
            "--debug",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(_PROJECT_ROOT),
    )
    if result.returncode != 0:
        # Print stdout/stderr for debugging
        print("=== STDOUT ===")
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        print("=== STDERR ===")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
    assert result.returncode == 0, (
        f"train_3d.py --debug exited with code {result.returncode}"
    )


@pytest.mark.skipif(bool(_MISSING_DEPS), reason=_skip_reason)
def test_train3d_debug_creates_checkpoint():
    """
    After a --debug run, at least one checkpoint file should exist
    in the checkpoint_dir configured in the YAML.
    """
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        # Override checkpoint_dir via a temporary config
        cfg_path = _PROJECT_ROOT / "config" / "anyup3d_train.yaml"
        cfg_text = cfg_path.read_text()
        # Replace checkpoint_dir line
        patched = cfg_text.replace(
            "checkpoint_dir: checkpoints/anyup3d/",
            f"checkpoint_dir: {tmpdir}/",
        )
        patched_cfg = Path(tmpdir) / "test_config.yaml"
        patched_cfg.write_text(patched)

        result = subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "train_3d.py"),
                "--config",
                str(patched_cfg),
                "--debug",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(_PROJECT_ROOT),
        )
        if result.returncode != 0:
            pytest.skip(f"train_3d.py failed (rc={result.returncode}), skipping checkpoint check")

        pth_files = list(Path(tmpdir).rglob("*.pth"))
        assert len(pth_files) > 0, (
            f"No .pth checkpoint files found in {tmpdir} after --debug run"
        )
