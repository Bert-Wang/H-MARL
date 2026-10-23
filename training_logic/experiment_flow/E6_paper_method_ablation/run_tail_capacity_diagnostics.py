from __future__ import annotations

"""Frozen-scheduler tail-latency and capacity diagnostics for reviewer issue 7.

No scheduler or predictor is trained or updated.  The five final schedulers are
replayed deterministically on the common PaperMethodEnvironment.  A fixed,
policy-independent horizon is then applied to the completed trajectory to
report:

* mean, median, and P95 task latency;
* task activation/completion coverage at the horizon;
* workflows unfinished at the horizon; and
* demand-weighted CPU utilization per node and per tier.

The default horizon is the maximum workflow deadline in the episode.  Because
deadlines come from the policy-independent reference schedule, every scheduler
is evaluated against the same horizon for the same environment seed.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent


def locate_training_logic_root() -> Path:
    override = os.environ.get("TRAINING_LOGIC_ROOT")
    candidates: List[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    for ancestor in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        candidates.extend((ancestor, ancestor / "training_logic"))
    checked: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        checked.append(key)
        if (
            (resolved / "03_methods" / "paper_method_hmarl" / "environment.py").is_file()
            and (
                resolved
                / "04_experiment_flow"
                / "E8_slack_predictor_optimization"
                / "run_frozen_drl_calibration.py"
            ).is_file()
        ):
            return resolved
    raise FileNotFoundError(
        "Cannot locate training_logic. Checked:\n  - "
        + "\n  - ".join(checked)
        + "\nSet TRAINING_LOGIC_ROOT to the Linux training_logic directory."
    )


TRAINING_LOGIC_ROOT = locate_training_logic_root()
METHOD_ROOT = TRAINING_LOGIC_ROOT / "03_methods"
E8_FLOW_ROOT = (
    TRAINING_LOGIC_ROOT
    / "04_experiment_flow"
    / "E8_slack_predictor_optimization"
)
for import_root in (METHOD_ROOT, E8_FLOW_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from paper_method_hmarl.config import ExperimentConfig, MethodSwitches  # noqa: E402
from paper_method_hmarl.trainer import (  # noqa: E402
    PolicyRuntime,
    build_environment,
    load_predictor,
    resolve_device,
    safe_torch_load,
    set_random_seed,
)
from run_frozen_drl_calibration import (  # noqa: E402
    SOURCE_LABELS,
    _BaselinePolicy,
    _PWHSA,
)


RESULTS_ROOT = TRAINING_LOGIC_ROOT / "06_results_archive"
MODEL_ROOT = RESULTS_ROOT / "models"
DEFAULT_PREDICTOR = (
    RESULTS_ROOT
    / "E6_paper_method_ablation"
    / "shared"
    / "slack_histgbrt_remaining_v2.joblib"
)
DEFAULT_HMARL_CHECKPOINT = (
    RESULTS_ROOT
    / "E6_paper_method_ablation"
    / "runs"
    / "ablation_full_100users_seed42_deadline_v2"
    / "best_model.pt"
)
DEFAULT_CHECKPOINT_TEMPLATES = {
    "h_marl": str(DEFAULT_HMARL_CHECKPOINT),
    "hyperjet": str(
        MODEL_ROOT / "hyperjet" / "best_model_hyperjet_e4_seed{seed}.pth"
    ),
    "maddpg_ra": str(
        MODEL_ROOT / "maddpg" / "best_model_maddpg_e4_seed{seed}.pth"
    ),
    "a2c_drl": str(
        MODEL_ROOT / "maa2c" / "best_model_maa2c_e4_seed{seed}.pth"
    ),
}
DEFAULT_OUTPUT = (
    RESULTS_ROOT
    / "E6_paper_method_ablation"
    / "tail_capacity_diagnostics"
    / "main_100sources"
)

EPISODE_METRICS = (
    "task_latency_mean_s",
    "task_latency_median_s",
    "task_latency_p95_s",
    "task_latency_max_s",
    "completed_by_horizon_task_latency_mean_s",
    "completed_by_horizon_task_latency_p95_s",
    "horizon_observed_task_time_mean_s",
    "horizon_observed_task_time_p95_s",
    "task_activation_fraction_at_horizon",
    "task_completion_fraction_at_horizon",
    "activated_task_incomplete_fraction_at_horizon",
    "workflow_completion_mean_s",
    "workflow_completion_p95_s",
    "workflow_incomplete_fraction_at_horizon",
    "cpu_utilization_overall",
    "node_cpu_utilization_mean",
    "node_cpu_utilization_p95",
    "node_cpu_utilization_max",
    "node_busy_utilization_mean",
    "node_busy_utilization_p95",
    "pi_cpu_utilization_mean",
    "edge_cpu_utilization_mean",
    "cloud_cpu_utilization_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay five frozen schedulers and report task-latency tails, "
            "horizon completion coverage, and node CPU utilization."
        )
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=tuple(SOURCE_LABELS),
        default=list(SOURCE_LABELS),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument(
        "--checkpoint-seed",
        type=int,
        choices=(42,),
        default=42,
        help=(
            "Training seed of the fixed frozen checkpoints. Evaluation seeds "
            "remain controlled by --seeds."
        ),
    )
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--load", type=int, choices=(50, 100, 150), default=100)
    parser.add_argument("--templates", type=int, nargs="+", default=[4])
    parser.add_argument("--deadline-factor", type=float, default=1.10)
    parser.add_argument(
        "--horizon-mode",
        choices=("max_deadline", "reference_makespan", "fixed"),
        default="max_deadline",
    )
    parser.add_argument(
        "--horizon-seconds",
        type=float,
        default=None,
        help="Required only when --horizon-mode fixed is selected.",
    )
    parser.add_argument("--predictor", type=Path, default=DEFAULT_PREDICTOR)
    parser.add_argument(
        "--hmarl-checkpoint-template",
        default=DEFAULT_CHECKPOINT_TEMPLATES["h_marl"],
    )
    parser.add_argument(
        "--hyperjet-checkpoint-template",
        default=DEFAULT_CHECKPOINT_TEMPLATES["hyperjet"],
    )
    parser.add_argument(
        "--maddpg-checkpoint-template",
        default=DEFAULT_CHECKPOINT_TEMPLATES["maddpg_ra"],
    )
    parser.add_argument(
        "--a2c-checkpoint-template",
        default=DEFAULT_CHECKPOINT_TEMPLATES["a2c_drl"],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=None,
        help="Debug-only truncation. Omit for publication-ready metrics.",
    )
    parser.add_argument("--write-task-csv", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.seeds:
        raise ValueError("At least one seed is required.")
    if args.episodes_per_seed <= 0:
        raise ValueError("--episodes-per-seed must be positive.")
    if not args.templates or any(item not in range(5) for item in args.templates):
        raise ValueError("--templates must contain IDs in [0, 4].")
    if args.deadline_factor < 1.0:
        raise ValueError("--deadline-factor must be at least 1.0.")
    if args.horizon_mode == "fixed":
        if args.horizon_seconds is None or args.horizon_seconds <= 0:
            raise ValueError(
                "--horizon-seconds must be positive when --horizon-mode fixed."
            )
    if args.max_decisions is not None and args.max_decisions <= 0:
        raise ValueError("--max-decisions must be positive when provided.")
    if args.load != 100 and any(source != "pwhsa" for source in args.sources):
        raise ValueError(
            "The available learned-policy checkpoints use the 100-source state "
            "dimension. Use --load 100, or run PWSHA alone for debug loads."
        )


def checkpoint_templates(args: argparse.Namespace) -> Dict[str, str | None]:
    return {
        "h_marl": args.hmarl_checkpoint_template,
        "hyperjet": args.hyperjet_checkpoint_template,
        "pwhsa": None,
        "maddpg_ra": args.maddpg_checkpoint_template,
        "a2c_drl": args.a2c_checkpoint_template,
    }


def resolve_checkpoint(template: str | None, seed: int) -> Path | None:
    if template is None:
        return None
    path = Path(template.format(seed=seed)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Frozen checkpoint not found: {path}")
    return path


def make_config(
    *, args: argparse.Namespace, source: str, episode_seed: int
) -> ExperimentConfig:
    uses_predictor = source == "h_marl"
    return ExperimentConfig(
        implementation_version="e6-deadline-v2",
        variant="full",
        switches=MethodSwitches(True, uses_predictor, True, True),
        num_users=args.load,
        seed=episode_seed,
        deadline_factor=args.deadline_factor,
        train_templates=(0, 1, 2, 3),
        test_templates=tuple(sorted(set(args.templates))),
        device=args.device,
        slack_model_path=str(args.predictor.resolve()) if uses_predictor else None,
    )


def build_frozen_policy(source: str, checkpoint: Path | None, env, config, device):
    if source == "h_marl":
        if checkpoint is None:
            raise ValueError("H-MARL requires a checkpoint.")
        runtime = PolicyRuntime(config, len(env._get_state()), env.tier_sizes, device)
        payload = safe_torch_load(checkpoint, device)
        if "runtime" not in payload:
            raise ValueError(f"Not a paper_method_hmarl checkpoint: {checkpoint}")
        runtime.load_state_dict(payload["runtime"], load_optimizers=False)
        for module in runtime.policies.values():
            module.eval()
        runtime.critic.eval()
        return runtime
    if source == "pwhsa":
        return _PWHSA(env)
    if checkpoint is None:
        raise ValueError(f"{source} requires a checkpoint.")
    return _BaselinePolicy(source, checkpoint, env, device)


def select_action(source: str, policy, env, state, masks, device):
    if source == "h_marl":
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            tier, node, _ = policy.select_action(
                state_tensor, masks, deterministic=True
            )
        return int(tier), int(node)
    tier, node = policy.select(env, state, masks)
    return int(tier), int(node)


def choose_horizon(args: argparse.Namespace, env) -> float:
    if args.horizon_mode == "fixed":
        return float(args.horizon_seconds)
    if args.horizon_mode == "reference_makespan":
        return float(max(env.workflow_reference_completions.values()))
    return float(max(env.workflow_deadlines.values()))


def task_activation_time(env, task_id: str) -> float:
    task = env.task_map[task_id]
    if not task.parents:
        return 0.0
    parent_ends = [
        float(env.task_infos[parent]["end"])
        for parent in task.parents
        if parent in env.task_infos
    ]
    if len(parent_ends) != len(task.parents):
        return float("nan")
    return float(max(parent_ends))


def percentile(values: Sequence[float], q: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.percentile(array, q)) if array.size else float("nan")


def safe_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()) if array.size else float("nan")


def compute_task_rows(env, horizon: float) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for task_id, task in env.task_map.items():
        activation = task_activation_time(env, task_id)
        info = env.task_infos.get(task_id)
        start = float(info["start"]) if info is not None else float("nan")
        finish = float(info["end"]) if info is not None else float("nan")
        activated = bool(np.isfinite(activation) and activation <= horizon + 1e-9)
        completed = bool(np.isfinite(finish) and finish <= horizon + 1e-9)
        eventual_latency = (
            finish - activation
            if np.isfinite(finish) and np.isfinite(activation)
            else float("nan")
        )
        censored_latency = (
            max(0.0, min(finish, horizon) - activation)
            if activated and np.isfinite(finish)
            else (max(0.0, horizon - activation) if activated else float("nan"))
        )
        rows.append(
            {
                "task_id": task_id,
                "workflow_id": task.workflow_id,
                "privacy_level": int(task.privacy_level),
                "activation_time": activation,
                "start_time": start,
                "finish_time": finish,
                "node": str(info["node"]) if info is not None else "",
                "activated_at_horizon": activated,
                "completed_at_horizon": completed,
                "task_latency_s": eventual_latency,
                "completed_by_horizon_task_latency_s": (
                    eventual_latency if completed else float("nan")
                ),
                "horizon_observed_task_time_s": censored_latency,
            }
        )
    return rows


def compute_workflow_rows(env, horizon: float) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for workflow_id, task_ids in env.workflow_tasks.items():
        ends = [
            float(env.task_infos[task_id]["end"])
            for task_id in task_ids
            if task_id in env.task_infos
        ]
        completion = float(max(ends)) if len(ends) == len(task_ids) else float("nan")
        rows.append(
            {
                "workflow_id": workflow_id,
                "completion_time": completion,
                "deadline": float(env.workflow_deadlines[workflow_id]),
                "completed_at_horizon": bool(
                    np.isfinite(completion) and completion <= horizon + 1e-9
                ),
                "eventually_completed": len(ends) == len(task_ids),
            }
        )
    return rows


def compute_node_rows(env, horizon: float) -> List[Dict[str, object]]:
    node_core_seconds = defaultdict(float)
    node_busy_seconds = defaultdict(float)
    for task_id, info in env.task_infos.items():
        start = float(info["start"])
        finish = float(info["end"])
        overlap = max(0.0, min(finish, horizon) - max(start, 0.0))
        if overlap <= 0:
            continue
        node = str(info["node"])
        node_busy_seconds[node] += overlap
        node_core_seconds[node] += overlap * float(env.task_map[task_id].core_count)

    rows: List[Dict[str, object]] = []
    for node in env.all_nodes:
        capacity = float(env.node_specs[node]["cpu"])
        denominator = max(capacity * horizon, 1e-12)
        rows.append(
            {
                "node": node,
                "tier": str(env.node_specs[node]["type"]),
                "cpu_capacity_cores": capacity,
                "cpu_core_seconds": float(node_core_seconds[node]),
                "busy_seconds": float(node_busy_seconds[node]),
                "cpu_utilization": float(node_core_seconds[node] / denominator),
                "busy_time_utilization": float(
                    node_busy_seconds[node] / max(horizon, 1e-12)
                ),
            }
        )
    return rows


def summarize_episode(
    *,
    source: str,
    base_seed: int,
    episode: int,
    episode_seed: int,
    checkpoint_seed: int | None,
    checkpoint: Path | None,
    horizon: float,
    horizon_mode: str,
    env,
    task_rows: Sequence[Mapping[str, object]],
    workflow_rows: Sequence[Mapping[str, object]],
    node_rows: Sequence[Mapping[str, object]],
    truncated: bool,
) -> Dict[str, object]:
    all_task_latencies = [
        float(row["task_latency_s"])
        for row in task_rows
        if np.isfinite(float(row["task_latency_s"]))
    ]
    completed_by_horizon_latencies = [
        float(row["completed_by_horizon_task_latency_s"])
        for row in task_rows
        if np.isfinite(float(row["completed_by_horizon_task_latency_s"]))
    ]
    horizon_observed_times = [
        float(row["horizon_observed_task_time_s"])
        for row in task_rows
        if np.isfinite(float(row["horizon_observed_task_time_s"]))
    ]
    activated = sum(bool(row["activated_at_horizon"]) for row in task_rows)
    completed = sum(bool(row["completed_at_horizon"]) for row in task_rows)
    incomplete_activated = max(activated - completed, 0)
    workflow_completions = [
        float(row["completion_time"])
        for row in workflow_rows
        if np.isfinite(float(row["completion_time"]))
    ]
    incomplete_workflows = sum(
        not bool(row["completed_at_horizon"]) for row in workflow_rows
    )
    cpu_values = [float(row["cpu_utilization"]) for row in node_rows]
    busy_values = [float(row["busy_time_utilization"]) for row in node_rows]
    total_core_seconds = sum(float(row["cpu_core_seconds"]) for row in node_rows)
    total_capacity_seconds = sum(
        float(row["cpu_capacity_cores"]) * horizon for row in node_rows
    )

    summary: Dict[str, object] = {
        "source": source,
        "source_label": SOURCE_LABELS[source],
        "base_seed": base_seed,
        "checkpoint_seed": checkpoint_seed,
        "episode": episode,
        "episode_seed": episode_seed,
        "checkpoint": str(checkpoint) if checkpoint is not None else "deterministic PWSHA",
        "horizon_mode": horizon_mode,
        "horizon_s": horizon,
        "trajectory_makespan_s": float(env.current_time),
        "decisions": len(env.assigned_tasks),
        "tasks": len(task_rows),
        "workflows": len(workflow_rows),
        "truncated": truncated,
        "publication_ready": not truncated,
        "task_latency_denominator": "all tasks in the complete frozen-policy replay",
        "task_latency_mean_s": safe_mean(all_task_latencies),
        "task_latency_median_s": percentile(all_task_latencies, 50),
        "task_latency_p95_s": percentile(all_task_latencies, 95),
        "task_latency_max_s": percentile(all_task_latencies, 100),
        "completed_by_horizon_task_latency_mean_s": safe_mean(
            completed_by_horizon_latencies
        ),
        "completed_by_horizon_task_latency_p95_s": percentile(
            completed_by_horizon_latencies, 95
        ),
        "horizon_observed_task_time_mean_s": safe_mean(horizon_observed_times),
        "horizon_observed_task_time_p95_s": percentile(
            horizon_observed_times, 95
        ),
        "task_activation_fraction_at_horizon": activated / max(len(task_rows), 1),
        "task_completion_fraction_at_horizon": completed / max(len(task_rows), 1),
        "activated_task_incomplete_fraction_at_horizon": incomplete_activated
        / max(activated, 1),
        "workflow_completion_mean_s": safe_mean(workflow_completions),
        "workflow_completion_p95_s": percentile(workflow_completions, 95),
        "workflow_incomplete_fraction_at_horizon": incomplete_workflows
        / max(len(workflow_rows), 1),
        "cpu_utilization_overall": total_core_seconds
        / max(total_capacity_seconds, 1e-12),
        "node_cpu_utilization_mean": safe_mean(cpu_values),
        "node_cpu_utilization_p95": percentile(cpu_values, 95),
        "node_cpu_utilization_max": percentile(cpu_values, 100),
        "node_busy_utilization_mean": safe_mean(busy_values),
        "node_busy_utilization_p95": percentile(busy_values, 95),
    }
    for tier in ("pi", "edge", "cloud"):
        tier_values = [
            float(row["cpu_utilization"])
            for row in node_rows
            if str(row["tier"]) == tier
        ]
        summary[f"{tier}_cpu_utilization_mean"] = safe_mean(tier_values)
        summary[f"{tier}_cpu_utilization_p95"] = percentile(tier_values, 95)
    utilization_fields = (
        "cpu_utilization_overall",
        "node_cpu_utilization_mean",
        "node_cpu_utilization_p95",
        "node_cpu_utilization_max",
    )
    for field in utilization_fields:
        value = float(summary[field])
        if not (-1e-9 <= value <= 1.0 + 1e-9):
            raise RuntimeError(
                f"Simulated reserved-core utilization is outside [0, 1]: "
                f"{field}={value}"
            )
    return summary


def replay_episode(
    *,
    args: argparse.Namespace,
    source: str,
    base_seed: int,
    episode: int,
    checkpoint: Path | None,
    device: torch.device,
) -> tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    episode_seed = int(base_seed + episode)
    set_random_seed(episode_seed)
    config = make_config(args=args, source=source, episode_seed=episode_seed)
    predictor = load_predictor(config)
    env = build_environment(config, predictor, evaluation=True)
    state = env.reset()
    policy = build_frozen_policy(source, checkpoint, env, config, device)
    done = False
    decisions = 0
    while not done:
        masks = env._get_action_mask()
        tier, node = select_action(source, policy, env, state, masks, device)
        state, _, done, _, _ = env.step(tier, node)
        decisions += 1
        if args.progress_every and decisions % args.progress_every == 0:
            print(
                f"{SOURCE_LABELS[source]} seed={base_seed} episode={episode + 1} "
                f"decisions={decisions}/{len(env.task_map)}"
            )
        if args.max_decisions is not None and decisions >= args.max_decisions:
            break

    truncated = not done
    horizon = choose_horizon(args, env)
    task_rows = compute_task_rows(env, horizon)
    workflow_rows = compute_workflow_rows(env, horizon)
    node_rows = compute_node_rows(env, horizon)
    summary = summarize_episode(
        source=source,
        base_seed=base_seed,
        episode=episode + 1,
        episode_seed=episode_seed,
        checkpoint_seed=(None if source == "pwhsa" else args.checkpoint_seed),
        checkpoint=checkpoint,
        horizon=horizon,
        horizon_mode=args.horizon_mode,
        env=env,
        task_rows=task_rows,
        workflow_rows=workflow_rows,
        node_rows=node_rows,
        truncated=truncated,
    )
    return summary, task_rows, workflow_rows, node_rows


def aggregate_sources(
    rows: Sequence[Mapping[str, object]],
    task_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for source in SOURCE_LABELS:
        selected = [row for row in rows if row["source"] == source]
        if not selected:
            continue
        summary: Dict[str, object] = {
            "source": source,
            "source_label": SOURCE_LABELS[source],
            "episodes": len(selected),
            "publication_ready": all(bool(row["publication_ready"]) for row in selected),
        }
        source_tasks = [row for row in task_rows if row["source"] == source]
        pooled_latencies = [
            float(row["task_latency_s"])
            for row in source_tasks
            if np.isfinite(float(row["task_latency_s"]))
        ]
        summary.update(
            {
                "pooled_task_count": len(pooled_latencies),
                "pooled_task_latency_mean_s": safe_mean(pooled_latencies),
                "pooled_task_latency_p95_s": percentile(pooled_latencies, 95),
            }
        )
        for metric in EPISODE_METRICS:
            values = np.asarray(
                [float(row[metric]) for row in selected], dtype=np.float64
            )
            finite = values[np.isfinite(values)]
            summary[f"{metric}_mean"] = (
                float(finite.mean()) if finite.size else float("nan")
            )
            summary[f"{metric}_std"] = (
                float(finite.std(ddof=1)) if finite.size > 1 else 0.0
            )
            summary[f"{metric}_min"] = (
                float(finite.min()) if finite.size else float("nan")
            )
            summary[f"{metric}_max"] = (
                float(finite.max()) if finite.size else float("nan")
            )
        output.append(summary)
    return output


def build_tradeoff_rows(
    episode_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    hmarl_by_episode = {
        (int(row["base_seed"]), int(row["episode"]), int(row["episode_seed"])): row
        for row in episode_rows
        if row["source"] == "h_marl"
    }
    rows: List[Dict[str, object]] = []
    for baseline in episode_rows:
        if baseline["source"] == "h_marl":
            continue
        key = (
            int(baseline["base_seed"]),
            int(baseline["episode"]),
            int(baseline["episode_seed"]),
        )
        hmarl = hmarl_by_episode.get(key)
        if hmarl is None:
            continue
        mean_delta = float(hmarl["task_latency_mean_s"]) - float(
            baseline["task_latency_mean_s"]
        )
        p95_delta = float(hmarl["task_latency_p95_s"]) - float(
            baseline["task_latency_p95_s"]
        )
        mean_improved = mean_delta < 0.0
        p95_worsened = p95_delta > 0.0
        rows.append(
            {
                "baseline_source": baseline["source"],
                "baseline_label": baseline["source_label"],
                "base_seed": baseline["base_seed"],
                "episode": baseline["episode"],
                "episode_seed": baseline["episode_seed"],
                "horizon_s": baseline["horizon_s"],
                "hmarl_task_latency_mean_s": hmarl["task_latency_mean_s"],
                "baseline_task_latency_mean_s": baseline["task_latency_mean_s"],
                "mean_delta_hmarl_minus_baseline_s": mean_delta,
                "mean_improved": mean_improved,
                "hmarl_task_latency_p95_s": hmarl["task_latency_p95_s"],
                "baseline_task_latency_p95_s": baseline["task_latency_p95_s"],
                "p95_delta_hmarl_minus_baseline_s": p95_delta,
                "p95_worsened": p95_worsened,
                "mean_improved_but_tail_worsened": (
                    mean_improved and p95_worsened
                ),
                "hmarl_workflow_incomplete_fraction": hmarl[
                    "workflow_incomplete_fraction_at_horizon"
                ],
                "baseline_workflow_incomplete_fraction": baseline[
                    "workflow_incomplete_fraction_at_horizon"
                ],
                "hmarl_cpu_utilization": hmarl["cpu_utilization_overall"],
                "baseline_cpu_utilization": baseline["cpu_utilization_overall"],
            }
        )
    return rows


def aggregate_tradeoffs(
    rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for baseline_source in SOURCE_LABELS:
        if baseline_source == "h_marl":
            continue
        selected = [
            row for row in rows if row["baseline_source"] == baseline_source
        ]
        if not selected:
            continue
        mean_deltas = [
            float(row["mean_delta_hmarl_minus_baseline_s"]) for row in selected
        ]
        p95_deltas = [
            float(row["p95_delta_hmarl_minus_baseline_s"]) for row in selected
        ]
        tradeoff_count = sum(
            bool(row["mean_improved_but_tail_worsened"]) for row in selected
        )
        output.append(
            {
                "baseline_source": baseline_source,
                "baseline_label": SOURCE_LABELS[baseline_source],
                "paired_episodes": len(selected),
                "mean_delta_hmarl_minus_baseline_s_mean": safe_mean(mean_deltas),
                "mean_delta_hmarl_minus_baseline_s_min": percentile(
                    mean_deltas, 0
                ),
                "mean_delta_hmarl_minus_baseline_s_max": percentile(
                    mean_deltas, 100
                ),
                "p95_delta_hmarl_minus_baseline_s_mean": safe_mean(p95_deltas),
                "p95_delta_hmarl_minus_baseline_s_min": percentile(p95_deltas, 0),
                "p95_delta_hmarl_minus_baseline_s_max": percentile(
                    p95_deltas, 100
                ),
                "mean_improved_episode_count": sum(
                    bool(row["mean_improved"]) for row in selected
                ),
                "p95_worsened_episode_count": sum(
                    bool(row["p95_worsened"]) for row in selected
                ),
                "mean_improved_but_tail_worsened_count": tradeoff_count,
                "mean_improved_but_tail_worsened_fraction": tradeoff_count
                / len(selected),
                "mean_improved_but_tail_worsened_any": tradeoff_count > 0,
            }
        )
    return output


def json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.predictor = args.predictor.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if "h_marl" in args.sources and not args.predictor.exists():
        raise FileNotFoundError(f"H-MARL predictor not found: {args.predictor}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    templates = checkpoint_templates(args)

    episode_rows: List[Dict[str, object]] = []
    node_rows_all: List[Dict[str, object]] = []
    workflow_rows_all: List[Dict[str, object]] = []
    task_rows_all: List[Dict[str, object]] = []
    horizon_by_episode: Dict[tuple[int, int], float] = {}

    for source in args.sources:
        for seed in args.seeds:
            checkpoint = resolve_checkpoint(templates[source], args.checkpoint_seed)
            for episode in range(args.episodes_per_seed):
                print(
                    f"Replaying {SOURCE_LABELS[source]}: seed={seed}, "
                    f"episode={episode + 1}/{args.episodes_per_seed}"
                )
                summary, task_rows, workflow_rows, node_rows = replay_episode(
                    args=args,
                    source=source,
                    base_seed=int(seed),
                    episode=episode,
                    checkpoint=checkpoint,
                    device=device,
                )
                key = (int(summary["episode_seed"]), int(summary["episode"]))
                previous_horizon = horizon_by_episode.get(key)
                if previous_horizon is not None and not np.isclose(
                    previous_horizon,
                    float(summary["horizon_s"]),
                    rtol=1e-10,
                    atol=1e-10,
                ):
                    raise RuntimeError(
                        f"Policy-dependent horizon detected for episode {key}: "
                        f"{previous_horizon} vs {summary['horizon_s']}"
                    )
                horizon_by_episode[key] = float(summary["horizon_s"])
                episode_rows.append(summary)
                common = {
                    "source": source,
                    "source_label": SOURCE_LABELS[source],
                    "base_seed": int(seed),
                    "checkpoint_seed": (
                        None if source == "pwhsa" else args.checkpoint_seed
                    ),
                    "episode": episode + 1,
                    "episode_seed": int(summary["episode_seed"]),
                    "horizon_s": float(summary["horizon_s"]),
                }
                node_rows_all.extend({**common, **row} for row in node_rows)
                workflow_rows_all.extend({**common, **row} for row in workflow_rows)
                task_rows_all.extend({**common, **row} for row in task_rows)

    source_rows = aggregate_sources(episode_rows, task_rows_all)
    tradeoff_rows = build_tradeoff_rows(episode_rows)
    tradeoff_summary_rows = aggregate_tradeoffs(tradeoff_rows)
    write_csv(args.output_dir / "tail_capacity_episode_metrics.csv", episode_rows)
    write_csv(args.output_dir / "tail_capacity_source_summary.csv", source_rows)
    write_csv(args.output_dir / "tail_capacity_node_utilization.csv", node_rows_all)
    write_csv(args.output_dir / "tail_capacity_workflow_metrics.csv", workflow_rows_all)
    write_csv(args.output_dir / "tail_tradeoff_comparison.csv", tradeoff_rows)
    write_csv(args.output_dir / "tail_tradeoff_summary.csv", tradeoff_summary_rows)
    if args.write_task_csv:
        write_csv(args.output_dir / "tail_capacity_task_metrics.csv", task_rows_all)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": "deterministic frozen-scheduler replay; no training or parameter updates",
        "sources": args.sources,
        "source_labels": {source: SOURCE_LABELS[source] for source in args.sources},
        "seeds": args.seeds,
        "checkpoint_seed": args.checkpoint_seed,
        "episodes_per_seed": args.episodes_per_seed,
        "load": args.load,
        "templates": args.templates,
        "deadline_factor": args.deadline_factor,
        "horizon_mode": args.horizon_mode,
        "fixed_horizon_seconds": args.horizon_seconds,
        "max_decisions": args.max_decisions,
        "publication_ready": args.max_decisions is None
        and all(bool(row["publication_ready"]) for row in episode_rows),
        "checkpoint_templates": templates,
        "predictor": str(args.predictor),
        "resolved_device": str(device),
        "argv": sys.argv,
        "metric_definitions": {
            "task_latency_mean_s": "mean finish minus activation over all tasks in the complete replay",
            "task_latency_p95_s": "95th percentile finish minus activation over all tasks in the complete replay",
            "completed_by_horizon_task_latency_p95_s": "completed-only P95 before the fixed horizon; diagnostic only because it can have survivor bias",
            "horizon_observed_task_time_p95_s": "P95 of min(finish, horizon) minus activation among tasks activated by the horizon; an observed-time lower-bound diagnostic, not a censoring-adjusted estimator",
            "workflow_incomplete_fraction_at_horizon": "workflows not completed by the fixed horizon divided by all workflows",
            "cpu_utilization": "simulated reserved-core utilization: sum of task core demand times horizon-clipped execution duration divided by node CPU cores times horizon; not hardware telemetry",
            "busy_time_utilization": "horizon-clipped node busy seconds divided by horizon",
            "mean_improved_but_tail_worsened": "true when H-MARL has lower all-task mean latency but higher all-task P95 latency than the paired baseline episode",
        },
        "per_episode": episode_rows,
        "per_source": source_rows,
        "paired_tail_tradeoff": tradeoff_rows,
        "paired_tail_tradeoff_summary": tradeoff_summary_rows,
    }
    (args.output_dir / "tail_capacity_summary.json").write_text(
        json.dumps(json_safe(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nTail/capacity diagnostics completed.")
    print(f"Output directory: {args.output_dir}")
    print(json.dumps(json_safe(source_rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
