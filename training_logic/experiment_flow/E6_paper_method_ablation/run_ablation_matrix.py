from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


FLOW_DIR = Path(__file__).resolve().parent
TRAINING_LOGIC_ROOT = FLOW_DIR.parents[1]
METHOD_DIR = TRAINING_LOGIC_ROOT / "03_methods" / "paper_method_hmarl"
DEFAULT_PYTHON = Path(r"D:\Python_project\CNN\python.exe")
DEFAULT_OUTPUT_ROOT = (
    TRAINING_LOGIC_ROOT / "06_results_archive" / "E6_paper_method_ablation"
)
DEFAULT_VARIANTS = ("full", "wo_slack", "wo_mask", "wo_slack_wo_mask")
SLACK_VARIANTS = {"full", "wo_mask"}


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass
class Job:
    variant: str
    load: int
    seed: int
    run_tag: str
    run_dir: str
    log_path: str
    command: List[str]
    status: str = "pending"
    return_code: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    resumed: bool = False
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper-method 100-user slack/mask ablation matrix with isolated "
            "outputs and checkpoint resume."
        )
    )
    parser.add_argument("--loads", type=int, nargs="+", choices=(100,), default=[100])
    parser.add_argument(
        "--variants", nargs="+", choices=DEFAULT_VARIANTS, default=list(DEFAULT_VARIANTS)
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workload-jitter", type=float, default=0.05)
    parser.add_argument("--cross-workflow-edge-rate", type=float, default=0.30)
    parser.add_argument("--deadline-factor", type=float, default=1.10)
    parser.add_argument("--milp-strict", action="store_true")
    parser.add_argument("--milp-relax-top-k", type=int, default=3)
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rerun-sanity", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    invalid_loads = sorted(set(args.loads) - {100})
    if invalid_loads:
        raise ValueError(f"E6 ablation is fixed to 100 users; got {invalid_loads}")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if args.deadline_factor < 1.0:
        raise ValueError("deadline_factor must be at least 1.0")
    if args.milp_relax_top_k <= 0:
        raise ValueError("milp_relax_top_k must be positive")
    if not args.python_exe.is_file():
        raise FileNotFoundError(f"Python executable not found: {args.python_exe}")
    if not (METHOD_DIR / "train.py").is_file():
        raise FileNotFoundError(f"Method training entry not found: {METHOD_DIR / 'train.py'}")


def run_tag(variant: str, load: int, seed: int) -> str:
    return f"ablation_{variant}_{load}users_seed{seed}_deadline_v2"


def build_jobs(args: argparse.Namespace) -> List[Job]:
    runs_root = args.output_root / "runs"
    logs_root = args.output_root / "runner_logs"
    slack_model = args.output_root / "shared" / "slack_histgbrt_remaining_v2.joblib"
    jobs: List[Job] = []
    for load in args.loads:
        for variant in args.variants:
            for seed in args.seeds:
                tag = run_tag(variant, load, seed)
                job_run_dir = runs_root / tag
                command = [
                    str(args.python_exe),
                    str(METHOD_DIR / "train.py"),
                    "--variant",
                    variant,
                    "--num-users",
                    str(load),
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(args.epochs),
                    "--checkpoint-interval",
                    str(args.checkpoint_interval),
                    "--eval-episodes",
                    str(args.eval_episodes),
                    "--device",
                    args.device,
                    "--workload-jitter",
                    str(args.workload_jitter),
                    "--cross-workflow-edge-rate",
                    str(args.cross_workflow_edge_rate),
                    "--deadline-factor",
                    str(args.deadline_factor),
                    "--milp-relax-top-k",
                    str(args.milp_relax_top_k),
                    "--results-root",
                    str(runs_root),
                    "--run-tag",
                    tag,
                    "--slack-model",
                    str(slack_model),
                    "--no-auto-fit-slack",
                ]
                if args.max_steps is not None:
                    command.extend(("--max-steps", str(args.max_steps)))
                if args.milp_strict:
                    command.append("--milp-strict")
                checkpoint = job_run_dir / "last_checkpoint.pt"
                resumed = checkpoint.is_file()
                if resumed:
                    command.append("--resume")
                jobs.append(
                    Job(
                        variant=variant,
                        load=load,
                        seed=seed,
                        run_tag=tag,
                        run_dir=str(job_run_dir),
                        log_path=str(logs_root / f"{tag}.log"),
                        command=command,
                        resumed=resumed,
                    )
                )
    return jobs


def is_job_completed(job: Job, requested_epochs: int) -> bool:
    run_dir = Path(job.run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return int(summary.get("completed_epochs", 0)) >= requested_epochs
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def manifest_payload(args: argparse.Namespace, jobs: Iterable[Job]) -> Dict[str, object]:
    return {
        "updated_at": utc_now(),
        "python_exe": str(args.python_exe),
        "method_dir": str(METHOD_DIR),
        "output_root": str(args.output_root),
        "configuration": {
            "loads": args.loads,
            "variants": args.variants,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "checkpoint_interval": args.checkpoint_interval,
            "eval_episodes": args.eval_episodes,
            "max_steps": args.max_steps,
            "device": args.device,
            "workload_jitter": args.workload_jitter,
            "cross_workflow_edge_rate": args.cross_workflow_edge_rate,
            "deadline_factor": args.deadline_factor,
            "milp_strict": args.milp_strict,
            "milp_relax_top_k": args.milp_relax_top_k,
        },
        "jobs": [asdict(job) for job in jobs],
    }


def stream_process(
    command: List[str], log_path: Path, *, append: bool = False
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=TRAINING_LOGIC_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait(timeout=30)
            raise


def preflight(args: argparse.Namespace) -> None:
    command = [
        str(args.python_exe),
        "-c",
        (
            "import torch, scipy, sklearn, joblib, numpy, pandas; "
            "print('dependency preflight: OK'); "
            "print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
            "'gpus', torch.cuda.device_count())"
        ),
    ]
    result = subprocess.run(
        command,
        cwd=TRAINING_LOGIC_ROOT,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        raise RuntimeError(f"Dependency preflight failed:\n{result.stderr}")


def ensure_sanity(args: argparse.Namespace) -> None:
    marker = args.output_root / "shared" / "sanity_passed.json"
    source_mtime_ns = max(
        path.stat().st_mtime_ns
        for path in list(METHOD_DIR.glob("*.py")) + [Path(__file__).resolve()]
    )
    if marker.exists() and not args.rerun_sanity:
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            marker_current = (
                payload.get("python_exe") == str(args.python_exe)
                and int(payload.get("source_mtime_ns", 0)) == source_mtime_ns
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            marker_current = False
        if marker_current:
            print(f"[SANITY] already passed for current code/environment: {marker}")
            return
    log_path = args.output_root / "runner_logs" / "component_sanity.log"
    print("[SANITY] running real HistGBRT + MILP component check")
    return_code = stream_process(
        [str(args.python_exe), str(METHOD_DIR / "sanity_check.py"), "--steps", "5"],
        log_path,
    )
    if return_code != 0:
        raise RuntimeError(f"Component sanity failed; see {log_path}")
    write_json_atomic(
        marker,
        {
            "passed_at": utc_now(),
            "python_exe": str(args.python_exe),
            "source_mtime_ns": source_mtime_ns,
            "log": str(log_path),
        },
    )


def ensure_slack_model(args: argparse.Namespace) -> None:
    if not SLACK_VARIANTS.intersection(args.variants):
        return
    model_path = args.output_root / "shared" / "slack_histgbrt_remaining_v2.joblib"
    if model_path.exists():
        print(f"[SLACK] existing predictor: {model_path}")
        return
    log_path = args.output_root / "runner_logs" / "train_slack_predictor.log"
    print(f"[SLACK] fitting shared predictor: {model_path}")
    return_code = stream_process(
        [
            str(args.python_exe),
            str(METHOD_DIR / "train_slack_predictor.py"),
            "--output",
            str(model_path),
            "--loads",
            "100",
            "--seed",
            str(args.seeds[0]),
            "--templates",
            "0",
            "1",
            "2",
            "3",
        ],
        log_path,
    )
    if return_code != 0:
        raise RuntimeError(f"Slack predictor training failed; see {log_path}")


def aggregate_results(
    output_root: Path, jobs: Iterable[Job], target_epochs: int
) -> Path:
    rows: List[Dict[str, object]] = []
    for job in jobs:
        summary_path = Path(job.run_dir) / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        evaluation = summary.get("evaluation", {})
        row: Dict[str, object] = {
            "variant": job.variant,
            "load": job.load,
            "seed": job.seed,
            "run_tag": job.run_tag,
            "status": job.status,
            "completed_epochs": summary.get("completed_epochs"),
            "target_epochs": target_epochs,
            "best_epoch": summary.get("best_epoch"),
            "run_dir": job.run_dir,
        }
        for metric in (
            "avg_reward",
            "avg_latency",
            "avg_ddl",
            "avg_slack",
            "avg_wf_time",
            "ddl_rate",
            "mean_deadline",
            "mean_reference_completion",
            "avg_critical_slack",
            "avg_path_qoe",
            "avg_basic_actions",
            "avg_feasible_actions",
            "avg_policy_actions",
            "avg_pruned_actions",
            "avg_pruning_rate",
            "invalid_action_rate",
            "correction_rate",
            "basic_violation_rate",
            "qos_violation_rate",
            "critical_task_rate",
            "deadline_relaxation_rate",
        ):
            value = evaluation.get(metric, {})
            row[f"{metric}_mean"] = value.get("mean") if isinstance(value, dict) else None
            row[f"{metric}_std"] = value.get("std") if isinstance(value, dict) else None
        rows.append(row)
    output_path = output_root / "ablation_summary.csv"
    if not rows:
        return output_path
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    args = parse_args()
    args.python_exe = args.python_exe.resolve()
    args.output_root = args.output_root.resolve()
    validate_args(args)
    jobs = build_jobs(args)
    print(f"Prepared {len(jobs)} jobs")
    print(f"Python: {args.python_exe}")
    print(f"Dedicated output root: {args.output_root}")
    for job in jobs:
        completed = is_job_completed(job, args.epochs)
        state = "completed" if completed else ("resume" if job.resumed else "new")
        print(f"[{state:9}] {job.run_tag}")
        print("  " + subprocess.list2cmdline(job.command))
    if not args.execute:
        print("Dry run only. Add --execute to start the matrix.")
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "matrix_manifest.json"
    write_json_atomic(manifest_path, manifest_payload(args, jobs))
    preflight(args)
    ensure_sanity(args)
    ensure_slack_model(args)

    for index, job in enumerate(jobs, start=1):
        if is_job_completed(job, args.epochs):
            job.status = "skipped_completed"
            job.note = "summary.json already exists"
            job.finished_at = utc_now()
            print(f"[SKIP {index}/{len(jobs)}] {job.run_tag}: already completed")
            write_json_atomic(manifest_path, manifest_payload(args, jobs))
            continue

        job.status = "running"
        job.started_at = utc_now()
        print(
            f"[START {index}/{len(jobs)}] {job.run_tag} "
            f"({'resume' if job.resumed else 'fresh'})"
        )
        write_json_atomic(manifest_path, manifest_payload(args, jobs))
        try:
            return_code = stream_process(
                job.command, Path(job.log_path), append=job.resumed
            )
        except KeyboardInterrupt:
            job.status = "interrupted"
            job.note = "Interrupted by user; rerun the one-click script to resume."
            job.finished_at = utc_now()
            write_json_atomic(manifest_path, manifest_payload(args, jobs))
            raise
        job.return_code = return_code
        job.finished_at = utc_now()
        if return_code == 0:
            job.status = "completed"
            print(f"[DONE {index}/{len(jobs)}] {job.run_tag}")
        else:
            job.status = "failed"
            job.note = f"exit code {return_code}; see {job.log_path}"
            print(f"[FAILED {index}/{len(jobs)}] {job.run_tag}: {job.note}")
        write_json_atomic(manifest_path, manifest_payload(args, jobs))
        aggregate_results(args.output_root, jobs, args.epochs)
        if return_code != 0 and not args.continue_on_error:
            raise SystemExit(
                "Matrix stopped on the first failure. Fix the issue and rerun; "
                "existing checkpoints will resume automatically."
            )

    summary_path = aggregate_results(args.output_root, jobs, args.epochs)
    print("All requested ablation jobs are complete.")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
