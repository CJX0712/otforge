"""CLI smoke tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "otforge.cli", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def test_methods_subcommand_lists_the_registry():
    proc = _run_cli(["methods"])
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.strip().splitlines()
    assert "otfuse" in out
    assert "independent" in out


def test_run_writes_report_and_exits_zero(tmp_path: Path):
    out_file = tmp_path / "bench.json"
    proc = _run_cli(
        [
            "run",
            "--n-source",
            "24",
            "--n-target",
            "24",
            "--seeds",
            "0,1",
            "--anneal-steps",
            "8",
            "--output",
            str(out_file),
        ]
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["gate"]["passed"] is True
    assert set(data["gate"]["per_dataset"]) == {
        "gaussian_mix_1d",
        "point_cloud_2d",
        "grid_warp_2d",
        "domain_shift_2d",
    }


def test_run_exit_code_is_nonzero_when_the_gate_fails(tmp_path: Path):
    out_file = tmp_path / "bench.json"
    proc = _run_cli(
        [
            "run",
            "--n-source",
            "16",
            "--n-target",
            "16",
            "--seeds",
            "0",
            "--anneal-steps",
            "2",
            "--eps-final",
            "0.5",
            "--output",
            str(out_file),
        ]
    )
    assert proc.returncode == 1


def test_run_with_explicit_presets_and_methods():
    proc = _run_cli(
        [
            "run",
            "--n-source",
            "16",
            "--n-target",
            "16",
            "--seeds",
            "0",
            "--anneal-steps",
            "4",
            "--presets",
            "point_cloud_2d",
            "--methods",
            "independent,otfuse",
        ]
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "point_cloud_2d" in proc.stdout
    assert "gaussian_mix_1d" not in proc.stdout
