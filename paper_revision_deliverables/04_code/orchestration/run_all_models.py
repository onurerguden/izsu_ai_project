
"""
run_all_models.py

Runs the finalized model suite in a reproducible order:

1. Reactive Layer
2. Proactive Layer
3. Next-observation HF regression

All output paths are resolved from the project root, even when the script is
started with the VS Code Run button from inside the models directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def run_command(
    name: str,
    command: list[str],
    log_path: Path,
) -> dict[str, Any]:
    print("\n" + "=" * 72)
    print(f"[RUN] {name}")
    print(" ".join(command))
    print("=" * 72)

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        result.stdout
        + ("\nSTDERR:\n" + result.stderr if result.stderr else ""),
        encoding="utf-8",
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return {
        "name": name,
        "command": command,
        "return_code": result.returncode,
        "status": "SUCCESS" if result.returncode == 0 else "FAILED",
        "log_file": str(log_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all finalized İZSU model pipelines."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to izsu_features.csv",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root output directory",
    )
    parser.add_argument("--skip-reactive", action="store_true")
    parser.add_argument("--skip-proactive", action="store_true")
    parser.add_argument("--skip-regression", action="store_true")
    parser.add_argument(
        "--proactive-threshold",
        type=float,
        default=0.05,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    models_dir = Path(__file__).resolve().parent
    project_root = models_dir.parent

    input_path = (
        Path(args.input).resolve()
        if args.input
        else project_root / "data" / "data" / "izsu_features.csv"
    )
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else project_root / "outputs"
    )
    logs_dir = output_root / "logs"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Feature dosyası bulunamadı: {input_path}"
        )

    jobs = []
    if not args.skip_reactive:
        jobs.append(
            (
                "Reactive Layer",
                [
                    sys.executable,
                    str(models_dir / "classification_model.py"),
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_root / "reactive"),
                ],
                logs_dir / "reactive.log",
            )
        )

    if not args.skip_proactive:
        jobs.append(
            (
                "Proactive Layer",
                [
                    sys.executable,
                    str(models_dir / "proactive_trend_model.py"),
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_root / "proactive"),
                    "--threshold",
                    str(args.proactive_threshold),
                ],
                logs_dir / "proactive.log",
            )
        )

    if not args.skip_regression:
        jobs.append(
            (
                "Next-observation Regression",
                [
                    sys.executable,
                    str(
                        models_dir
                        / "hf_next_observation_regression.py"
                    ),
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_root / "regression"),
                ],
                logs_dir / "regression.log",
            )
        )

    manifest = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "project_root": str(project_root),
        "input": str(input_path),
        "output_root": str(output_root),
        "jobs": [],
    }

    failed = False
    for name, command, log_path in jobs:
        job_result = run_command(name, command, log_path)
        manifest["jobs"].append(job_result)
        if job_result["return_code"] != 0:
            failed = True
            break

    manifest["finished_at"] = datetime.now().isoformat(
        timespec="seconds"
    )
    manifest["status"] = "FAILED" if failed else "SUCCESS"

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run_all_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[MANIFEST] {manifest_path}")
    if failed:
        raise SystemExit(1)

    print("[SUCCESS] All requested model pipelines completed.")


if __name__ == "__main__":
    main()
