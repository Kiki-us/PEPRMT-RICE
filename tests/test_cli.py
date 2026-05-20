"""End-to-end test that the CLI runs a forward simulation on the example data."""

import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
EXAMPLE_CFG = REPO / "configs" / "example_forward.yaml"


@pytest.mark.slow
def test_cli_forward_run_smoke(tmp_path, monkeypatch):
    """`python -m peprmt_rice run --config example_forward.yaml` returns 0."""
    # Run inside the repo so relative paths in the example config resolve.
    monkeypatch.chdir(REPO / "configs")
    result = subprocess.run(
        [sys.executable, "-m", "peprmt_rice", "run", "--config", str(EXAMPLE_CFG)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "Done." in result.stdout


def test_cli_info_runs():
    result = subprocess.run(
        [sys.executable, "-m", "peprmt_rice", "info"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "PEPRMT-Rice" in result.stdout
