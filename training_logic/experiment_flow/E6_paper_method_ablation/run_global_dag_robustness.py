from __future__ import annotations

"""Reviewer issue 9: common-global-DAG bandwidth robustness test.

The five final schedulers are replayed without training under a nominal network
and configurable cross-tier bandwidth degradation scenarios.  By default the
experiment evaluates 90%, 70%, and 50% of nominal bandwidth for each of:

1. Pi--Cloud links;
2. Pi--Edge links;
3. Edge--Cloud links.

Every paired run uses the same global DAG, generated cross-workflow edges,
workflow deadlines, workload seed, task attributes, node set, and frozen
checkpoint.  The degradation is injected only after ``env.reset()`` so the
policy-independent reference deadlines remain identical across scenarios.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_tail_capacity_diagnostics import (  # noqa: E402
    DEFAULT_CHECKPOINT_TEMPLATES,
    DEFAULT_PREDICTOR,
    RESULTS_ROOT,
    SOURCE_LABELS,
    build_environment,
    build_frozen_policy,
    compute_node_rows,
    compute_task_rows,
    compute_workflow_rows,
    json_safe,
    load_predictor,
    make_config,
    percentile,
    resolve_checkpoint,
    resolve_device,
    safe_mean,
    select_action,
    set_random_seed,
    write_csv,
)


DEFAULT_OUTPUT_DIR = (
    RESULTS_ROOT
    / "E6_paper_method_ablation"
    / "global_dag_robustness"
    / "bandwidth_sweep_90_70_50"
)

CORE_METRICS = (
    "workflow_completion_mean_s",
    "workflow_completion_p95_s",
    "deadline_violation_rate",
    "workflow_deadline_slack_mean_s",
    "task_latency_mean_s",
    "task_latency_p95_s",
    "workflow_incomplete_fraction_at_horizon",
    "cpu_utilization_overall",
    "cloud_assignment_fraction",
    "trajectory_makespan_s",
)

LOWER_IS_BETTER = {
    "workflow_completion_mean_s": True,
    "workflow_completion_p95_s": True,
    "deadline_violation_rate": True,
    "workflow_deadline_slack_mean_s": False,
    "task_latency_mean_s": True,
    "task_latency_p95_s": True,
    "workflow_incomplete_fraction_at_horizon": True,
    "trajectory_makespan_s": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay five frozen schedulers on one common global DAG under "
            "nominal and multiple cross-tier bandwidth degradations."
        )
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=tuple(SOURCE_LABELS),
        default=list(SOURCE_LABELS),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument("--checkpoint-seed", type=int, choices=(42,), default=42)
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--load", type=int, choices=(50, 100, 150), default=100)
    parser.add_argument("--templates", type=int, nargs="+", default=[4])
    parser.add_argument("--deadline-factor", type=float, default=1.10)
    parser.add_argument(
        "--bandwidth-factors",
        type=float,
        nargs="+",
        default=[0.90, 0.70, 0.50],
        help="Bandwidth multipliers evaluated for every selected link group.",
    )
    parser.add_argument(
        "--link-groups",
        nargs="+",
        choices=("pi_cloud", "pi_edge", "edge_cloud"),
        default=["pi_cloud", "pi_edge", "edge_cloud"],
        help="Cross-tier link groups to degrade independently.",
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
    parser.add_argument(
        "--action-mask-protocol",
        choices=("native_system", "common_basic_mask"),
        default="native_system",
        help=(
            "native_system preserves H-MARL's predictor-conditioned MILP mask; "
            "common_basic_mask gives every scheduler the identical privacy/resource mask."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=None,
        help="Debug-only truncation. Omit for publication-ready results.",
    )
    parser.add_argument("--write-task-csv", action="store_true")
    parser.add_argument(
        "--require-publication-ready",
        action="store_true",
        help=(
            "Exit nonzero unless the run exactly matches the locked 150-episode "
            "publication protocol."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.seeds:
        raise ValueError("At least one evaluation seed is required.")
    if len(set(args.sources)) != len(args.sources):
        raise ValueError("--sources must not contain duplicates.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates.")
    if args.episodes_per_seed <= 0:
        raise ValueError("--episodes-per-seed must be positive.")
    if not args.templates or any(item not in range(5) for item in args.templates):
        raise ValueError("--templates must contain IDs in [0, 4].")
    if len(set(args.templates)) != len(args.templates):
        raise ValueError("--templates must not contain duplicates.")
    if args.deadline_factor < 1.0:
        raise ValueError("--deadline-factor must be at least 1.0.")
    if not args.bandwidth_factors:
        raise ValueError("At least one --bandwidth-factors value is required.")
    if any(not 0.0 < factor < 1.0 for factor in args.bandwidth_factors):
        raise ValueError(
            "Every --bandwidth-factors value must be strictly between 0 and 1."
        )
    if len(set(args.bandwidth_factors)) != len(args.bandwidth_factors):
        raise ValueError("--bandwidth-factors must not contain duplicates.")
    if len(set(args.link_groups)) != len(args.link_groups):
        raise ValueError("--link-groups must not contain duplicates.")
    if args.max_decisions is not None and args.max_decisions <= 0:
        raise ValueError("--max-decisions must be positive when provided.")
    if args.load != 100 and any(source != "pwhsa" for source in args.sources):
        raise ValueError(
            "The frozen learned policies use the 100-source action dimension. "
            "Use --load 100, or select PWSHA alone for a debug run."
        )


def checkpoint_templates(args: argparse.Namespace) -> Dict[str, str | None]:
    return {
        "h_marl": args.hmarl_checkpoint_template,
        "hyperjet": args.hyperjet_checkpoint_template,
        "pwhsa": None,
        "maddpg_ra": args.maddpg_checkpoint_template,
        "a2c_drl": args.a2c_checkpoint_template,
    }


def baseline_adaptation_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for source in args.sources:
        if source == "h_marl":
            training_protocol = (
                "PaperMethodEnvironment; train templates 0-3; test template 4; "
                "deadline factor 1.10"
            )
            state_adapter = "native PaperMethodEnvironment state"
        elif source == "pwhsa":
            training_protocol = "deterministic heuristic; no learned checkpoint"
            state_adapter = "common global-DAG task and node state"
        else:
            training_protocol = (
                "legacy HierarchicalDistributedSystemEnv checkpoint; evaluation "
                "is adapted to the common PaperMethodEnvironment global DAG"
            )
            state_adapter = (
                "legacy field order with privacy/3 and static deadline slack restored"
            )
        rows.append(
            {
                "source": source,
                "source_label": SOURCE_LABELS[source],
                "common_evaluation_environment": "PaperMethodEnvironment",
                "same_global_dag": True,
                "same_generated_cross_workflow_edges": True,
                "same_global_ready_queue": True,
                "same_task_runtime_data_privacy_attributes": True,
                "same_basic_privacy_resource_admissible_set": True,
                "policy_action_mask_protocol": args.action_mask_protocol,
                "same_policy_action_mask": (
                    args.action_mask_protocol == "common_basic_mask"
                ),
                "state_adapter": state_adapter,
                "checkpoint_training_protocol": training_protocol,
            }
        )
    return rows


def scenario_specs(args: argparse.Namespace) -> List[Tuple[str, str, float]]:
    scenarios: List[Tuple[str, str, float]] = [("nominal", "none", 1.0)]
    for link_group in args.link_groups:
        for factor in args.bandwidth_factors:
            scenarios.append(
                (f"{link_group}_bw_{factor:g}x", link_group, float(factor))
            )
    return scenarios


def global_edges(env) -> frozenset[Tuple[str, str]]:
    return frozenset(
        (parent, task_id)
        for task_id, task in env.task_map.items()
        for parent in task.parents
    )


def task_workload_signature(env) -> Dict[str, tuple]:
    return {
        task_id: (
            str(task.workflow_id),
            tuple(sorted(task.parents)),
            float(task.wf_runtime_sec),
            float(task.reference_cpu_time),
            float(task.reference_gpu_time),
            float(task.core_count),
            float(task.req_mem),
            float(task.req_gpu_mem),
            float(task.data_input_size),
            float(task.data_output_size),
            int(task.privacy_level),
            str(task.home_pi),
            tuple(task.allowed_nodes),
        )
        for task_id, task in env.task_map.items()
    }


def dag_audit(env) -> Dict[str, object]:
    edges = global_edges(env)
    cross_edges = sum(
        env.task_map[parent].workflow_id != env.task_map[child].workflow_id
        for parent, child in edges
    )
    root_tasks = sum(not task.parents for task in env.task_map.values())
    return {
        "tasks": len(env.task_map),
        "workflows": len(env.workflow_tasks),
        "total_global_edges": len(edges),
        "cross_workflow_edges": int(cross_edges),
        "cross_workflow_edge_fraction": cross_edges / max(len(edges), 1),
        "root_tasks": root_tasks,
        "global_dag_interface": (
            "common PaperMethodEnvironment task_map, global parent/child edges, "
            "ready queue, node state, and basic privacy/resource admissible set"
        ),
    }


def degrade_bandwidth(
    env, link_group: str, factor: float
) -> Dict[str, float]:
    affected_before: List[float] = []
    affected_after: List[float] = []
    if link_group == "none" or np.isclose(factor, 1.0):
        return {
            "affected_directed_links": 0,
            "bandwidth_before_mean_Bps": float("nan"),
            "bandwidth_after_mean_Bps": float("nan"),
        }
    target_pair = {
        "pi_cloud": frozenset(("pi", "cloud")),
        "pi_edge": frozenset(("pi", "edge")),
        "edge_cloud": frozenset(("edge", "cloud")),
    }[link_group]
    for source_index, source in enumerate(env.all_nodes):
        source_type = str(env.node_specs[source]["type"])
        for destination_index, destination in enumerate(env.all_nodes):
            destination_type = str(env.node_specs[destination]["type"])
            pair = frozenset((source_type, destination_type))
            if source_type == destination_type or pair != target_pair:
                continue
            before = float(env.bw_matrix[source_index, destination_index])
            if not np.isfinite(before):
                continue
            after = before * factor
            env.bw_matrix[source_index, destination_index] = after
            affected_before.append(before)
            affected_after.append(after)
    if not affected_before:
        raise RuntimeError(
            f"No finite directed links matched bandwidth group {link_group!r}."
        )
    return {
        "affected_directed_links": len(affected_before),
        "bandwidth_before_mean_Bps": safe_mean(affected_before),
        "bandwidth_after_mean_Bps": safe_mean(affected_after),
    }


def summarize_episode(
    *,
    source: str,
    scenario: str,
    link_group: str,
    bandwidth_factor: float,
    base_seed: int,
    episode: int,
    episode_seed: int,
    checkpoint_seed: int | None,
    checkpoint: Path | None,
    env,
    task_rows: Sequence[Mapping[str, object]],
    workflow_rows: Sequence[Mapping[str, object]],
    node_rows: Sequence[Mapping[str, object]],
    bandwidth_audit: Mapping[str, object],
    truncated: bool,
) -> Dict[str, object]:
    task_latencies = [
        float(row["task_latency_s"])
        for row in task_rows
        if np.isfinite(float(row["task_latency_s"]))
    ]
    completions = [
        float(row["completion_time"])
        for row in workflow_rows
        if np.isfinite(float(row["completion_time"]))
    ]
    slacks = [
        float(row["deadline"]) - float(row["completion_time"])
        for row in workflow_rows
        if np.isfinite(float(row["completion_time"]))
    ]
    violations = [value < 0.0 for value in slacks]
    incomplete = sum(
        not bool(row["completed_at_horizon"]) for row in workflow_rows
    )
    total_core_seconds = sum(float(row["cpu_core_seconds"]) for row in node_rows)
    horizon = float(max(env.workflow_deadlines.values()))
    total_capacity_seconds = sum(
        float(row["cpu_capacity_cores"]) * horizon for row in node_rows
    )
    cpu_utilization = total_core_seconds / max(total_capacity_seconds, 1e-12)
    if not (-1e-9 <= cpu_utilization <= 1.0 + 1e-9):
        raise RuntimeError(
            f"Reserved-core utilization outside [0, 1]: {cpu_utilization}"
        )
    assigned_nodes = [str(info["node"]) for info in env.task_infos.values()]
    cloud_assignments = sum(node.startswith("cloud_") for node in assigned_nodes)
    return {
        "source": source,
        "source_label": SOURCE_LABELS[source],
        "scenario": scenario,
        "link_group": link_group,
        "bandwidth_factor": bandwidth_factor,
        "base_seed": base_seed,
        "episode": episode,
        "episode_seed": episode_seed,
        "checkpoint_seed": checkpoint_seed,
        "checkpoint": str(checkpoint) if checkpoint else "deterministic PWSHA",
        "publication_ready": not truncated,
        "truncated": truncated,
        "horizon_s": horizon,
        "trajectory_makespan_s": float(env.current_time),
        "tasks": len(task_rows),
        "workflows": len(workflow_rows),
        "workflow_completion_mean_s": safe_mean(completions),
        "workflow_completion_p95_s": percentile(completions, 95),
        "deadline_violation_rate": safe_mean(violations),
        "workflow_deadline_slack_mean_s": safe_mean(slacks),
        "workflow_deadline_slack_p95_s": percentile(slacks, 95),
        "task_latency_mean_s": safe_mean(task_latencies),
        "task_latency_p95_s": percentile(task_latencies, 95),
        "workflow_incomplete_fraction_at_horizon": incomplete
        / max(len(workflow_rows), 1),
        "cpu_utilization_overall": cpu_utilization,
        "node_cpu_utilization_mean": safe_mean(
            [float(row["cpu_utilization"]) for row in node_rows]
        ),
        "node_cpu_utilization_p95": percentile(
            [float(row["cpu_utilization"]) for row in node_rows], 95
        ),
        "cloud_assignment_fraction": cloud_assignments
        / max(len(assigned_nodes), 1),
        **dict(bandwidth_audit),
    }


def baseline_compatible_state(source: str, env, state: np.ndarray) -> np.ndarray:
    """Restore the legacy neural-baseline feature semantics before reordering."""
    if source not in {"hyperjet", "maddpg_ra", "a2c_drl"}:
        return state
    adapted = np.asarray(state, dtype=np.float32).copy()
    task = env.current_task
    if task is None or adapted.shape[0] < 8:
        return adapted
    # Paper state position -2 uses privacy/2, while the legacy checkpoints were
    # trained with privacy/3.  Position -1 must be static deadline slack rather
    # than zero or the H-MARL predictor output.
    adapted[-2] = float(task.privacy_level) / 3.0
    adapted[-1] = (
        float(task.deadline)
        - float(env.current_time)
        - float(task.static_remaining_time)
    ) / 100.0
    return adapted


def policy_masks(args: argparse.Namespace, env):
    if args.action_mask_protocol == "common_basic_mask":
        if env.current_task is None:
            return env._mask_dict(np.ones(len(env.all_nodes), dtype=np.bool_))
        return env._mask_dict(env.basic_node_mask(env.current_task))
    return env._get_action_mask()


def replay_episode(
    *,
    args: argparse.Namespace,
    source: str,
    scenario: str,
    link_group: str,
    bandwidth_factor: float,
    base_seed: int,
    episode: int,
    checkpoint: Path | None,
    device: torch.device,
) -> tuple[
    Dict[str, object],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    frozenset[Tuple[str, str]],
    Dict[str, float],
    Dict[str, object],
    Dict[str, tuple],
]:
    episode_seed = int(base_seed + episode)
    set_random_seed(episode_seed)
    config = make_config(args=args, source=source, episode_seed=episode_seed)
    predictor = load_predictor(config)
    env = build_environment(config, predictor, evaluation=True)
    state = env.reset()

    edges = global_edges(env)
    workload_signature = task_workload_signature(env)
    deadlines = {
        workflow_id: float(value)
        for workflow_id, value in env.workflow_deadlines.items()
    }
    audit = dag_audit(env)
    bandwidth_audit = degrade_bandwidth(env, link_group, bandwidth_factor)
    if env.slack_predictor is not None:
        env.slack_predictor.invalidate_context_cache()
    # reset() returns the first observation before the degradation is injected.
    # Rebuild it so H-MARL's very first decision sees the degraded network too.
    state = env._get_state()
    policy = build_frozen_policy(source, checkpoint, env, config, device)

    done = False
    decisions = 0
    while not done:
        masks = policy_masks(args, env)
        policy_state = baseline_compatible_state(source, env, state)
        tier, node = select_action(
            source, policy, env, policy_state, masks, device
        )
        state, _, done, _, _ = env.step(tier, node)
        decisions += 1
        if args.progress_every and decisions % args.progress_every == 0:
            print(
                f"{SOURCE_LABELS[source]} {scenario} seed={base_seed} "
                f"episode={episode + 1} decisions={decisions}/{len(env.task_map)}"
            )
        if args.max_decisions is not None and decisions >= args.max_decisions:
            break

    truncated = not done
    horizon = float(max(deadlines.values()))
    task_rows = compute_task_rows(env, horizon)
    workflow_rows = compute_workflow_rows(env, horizon)
    node_rows = compute_node_rows(env, horizon)
    summary = summarize_episode(
        source=source,
        scenario=scenario,
        link_group=link_group,
        bandwidth_factor=bandwidth_factor,
        base_seed=base_seed,
        episode=episode + 1,
        episode_seed=episode_seed,
        checkpoint_seed=(None if source == "pwhsa" else args.checkpoint_seed),
        checkpoint=checkpoint,
        env=env,
        task_rows=task_rows,
        workflow_rows=workflow_rows,
        node_rows=node_rows,
        bandwidth_audit=bandwidth_audit,
        truncated=truncated,
    )
    summary.update(
        {
            "action_mask_protocol": args.action_mask_protocol,
            "policy_masks_are_identical_across_schedulers": (
                args.action_mask_protocol == "common_basic_mask"
            ),
            "predictor_uses_environment_context": bool(
                env.slack_predictor is not None
                and env.slack_predictor.uses_environment_context
            ),
            "baseline_state_adapter": (
                "legacy privacy/3 plus static deadline slack"
                if source in {"hyperjet", "maddpg_ra", "a2c_drl"}
                else "not applicable"
            ),
        }
    )
    return (
        summary,
        task_rows,
        workflow_rows,
        node_rows,
        edges,
        deadlines,
        audit,
        workload_signature,
    )


def add_scenario_ranks(rows: List[Dict[str, object]]) -> None:
    groups: Dict[Tuple[str, int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (str(row["scenario"]), int(row["base_seed"]), int(row["episode"]))
        ].append(row)
    for selected in groups.values():
        for metric in (
            "workflow_completion_mean_s",
            "deadline_violation_rate",
            "task_latency_p95_s",
        ):
            ordered = sorted(
                selected,
                key=lambda row: (
                    not np.isfinite(float(row[metric])),
                    float(row[metric])
                    if np.isfinite(float(row[metric]))
                    else float("inf"),
                    str(row["source"]),
                ),
            )
            previous_value = None
            current_rank = 0
            for position, row in enumerate(ordered, start=1):
                value = float(row[metric])
                if previous_value is None or not np.isclose(
                    value, previous_value, rtol=1e-10, atol=1e-10
                ):
                    current_rank = position
                    previous_value = value
                row[f"{metric}_rank"] = current_rank


def aggregate_source_scenarios(
    rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for source in SOURCE_LABELS:
        for scenario in sorted({str(row["scenario"]) for row in rows}):
            selected = [
                row
                for row in rows
                if row["source"] == source and row["scenario"] == scenario
            ]
            if not selected:
                continue
            summary: Dict[str, object] = {
                "source": source,
                "source_label": SOURCE_LABELS[source],
                "scenario": scenario,
                "link_group": selected[0]["link_group"],
                "bandwidth_factor": selected[0]["bandwidth_factor"],
                "episodes": len(selected),
                "publication_ready": all(
                    bool(row["publication_ready"]) for row in selected
                ),
            }
            for metric in CORE_METRICS:
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
            for metric in (
                "workflow_completion_mean_s",
                "deadline_violation_rate",
                "task_latency_p95_s",
            ):
                ranks = [float(row[f"{metric}_rank"]) for row in selected]
                summary[f"{metric}_rank_mean"] = safe_mean(ranks)
                summary[f"{metric}_rank_min"] = percentile(ranks, 0)
                summary[f"{metric}_rank_max"] = percentile(ranks, 100)
            output.append(summary)
    return output


def paired_degradation_rows(
    rows: Sequence[Mapping[str, object]], degraded_scenarios: Sequence[str]
) -> List[Dict[str, object]]:
    keyed = {
        (
            str(row["source"]),
            int(row["base_seed"]),
            int(row["episode"]),
            str(row["scenario"]),
        ): row
        for row in rows
    }
    output: List[Dict[str, object]] = []
    for source in SOURCE_LABELS:
        for base_seed in sorted({int(row["base_seed"]) for row in rows}):
            episodes = sorted(
                {
                    int(row["episode"])
                    for row in rows
                    if row["source"] == source and int(row["base_seed"]) == base_seed
                }
            )
            for episode in episodes:
                nominal = keyed.get((source, base_seed, episode, "nominal"))
                for degraded_scenario in degraded_scenarios:
                    degraded = keyed.get(
                        (source, base_seed, episode, degraded_scenario)
                    )
                    if nominal is None or degraded is None:
                        continue
                    paired: Dict[str, object] = {
                        "source": source,
                        "source_label": SOURCE_LABELS[source],
                        "base_seed": base_seed,
                        "episode": episode,
                        "episode_seed": nominal["episode_seed"],
                        "degraded_scenario": degraded_scenario,
                        "link_group": degraded["link_group"],
                        "bandwidth_factor": degraded["bandwidth_factor"],
                    }
                    for metric in CORE_METRICS:
                        nominal_value = float(nominal[metric])
                        degraded_value = float(degraded[metric])
                        delta = degraded_value - nominal_value
                        paired[f"nominal_{metric}"] = nominal_value
                        paired[f"degraded_{metric}"] = degraded_value
                        paired[f"delta_{metric}"] = delta
                        paired[f"relative_change_{metric}"] = (
                            delta / abs(nominal_value)
                            if abs(nominal_value) > 1e-12
                            else float("nan")
                        )
                    for metric in (
                        "workflow_completion_mean_s",
                        "deadline_violation_rate",
                        "task_latency_p95_s",
                    ):
                        paired[f"nominal_{metric}_rank"] = nominal[
                            f"{metric}_rank"
                        ]
                        paired[f"degraded_{metric}_rank"] = degraded[
                            f"{metric}_rank"
                        ]
                        paired[f"rank_change_{metric}"] = int(
                            degraded[f"{metric}_rank"]
                        ) - int(nominal[f"{metric}_rank"])
                    output.append(paired)
    return output


def hmarl_advantage_rows(
    rows: Sequence[Mapping[str, object]], degraded_scenarios: Sequence[str]
) -> List[Dict[str, object]]:
    keyed = {
        (
            str(row["source"]),
            int(row["base_seed"]),
            int(row["episode"]),
            str(row["scenario"]),
        ): row
        for row in rows
    }
    output: List[Dict[str, object]] = []
    compared_metrics = (
        "workflow_completion_mean_s",
        "deadline_violation_rate",
        "workflow_deadline_slack_mean_s",
        "task_latency_p95_s",
    )
    for baseline in SOURCE_LABELS:
        if baseline == "h_marl":
            continue
        keys = sorted(
            {
                (int(row["base_seed"]), int(row["episode"]))
                for row in rows
                if row["source"] == baseline
            }
        )
        for base_seed, episode in keys:
            for degraded_scenario in degraded_scenarios:
                for metric in compared_metrics:
                    h_nominal = keyed.get(
                        ("h_marl", base_seed, episode, "nominal")
                    )
                    b_nominal = keyed.get(
                        (baseline, base_seed, episode, "nominal")
                    )
                    h_degraded = keyed.get(
                        ("h_marl", base_seed, episode, degraded_scenario)
                    )
                    b_degraded = keyed.get(
                        (baseline, base_seed, episode, degraded_scenario)
                    )
                    if any(
                        item is None
                        for item in (
                            h_nominal,
                            b_nominal,
                            h_degraded,
                            b_degraded,
                        )
                    ):
                        continue
                    if LOWER_IS_BETTER[metric]:
                        nominal_advantage = float(b_nominal[metric]) - float(
                            h_nominal[metric]
                        )
                        degraded_advantage = float(b_degraded[metric]) - float(
                            h_degraded[metric]
                        )
                    else:
                        nominal_advantage = float(h_nominal[metric]) - float(
                            b_nominal[metric]
                        )
                        degraded_advantage = float(h_degraded[metric]) - float(
                            b_degraded[metric]
                        )
                    tolerance = 1e-9
                    if nominal_advantage > tolerance:
                        if degraded_advantage <= tolerance:
                            status = "vanished_or_reversed"
                        elif degraded_advantage > nominal_advantage + tolerance:
                            status = "expanded"
                        elif degraded_advantage < nominal_advantage - tolerance:
                            status = "shrunk"
                        else:
                            status = "stable"
                    elif degraded_advantage > tolerance:
                        status = "emerged"
                    else:
                        status = "no_positive_advantage"
                    output.append(
                        {
                            "baseline_source": baseline,
                            "baseline_label": SOURCE_LABELS[baseline],
                            "base_seed": base_seed,
                            "episode": episode,
                            "degraded_scenario": degraded_scenario,
                            "link_group": h_degraded["link_group"],
                            "bandwidth_factor": h_degraded[
                                "bandwidth_factor"
                            ],
                            "metric": metric,
                            "higher_advantage_is_better": True,
                            "nominal_hmarl_advantage": nominal_advantage,
                            "degraded_hmarl_advantage": degraded_advantage,
                            "advantage_change_degraded_minus_nominal": (
                                degraded_advantage - nominal_advantage
                            ),
                            "advantage_present_nominal": nominal_advantage > 0.0,
                            "advantage_present_degraded": degraded_advantage > 0.0,
                            "advantage_vanished_or_reversed": (
                                nominal_advantage > 0.0
                                and degraded_advantage <= 0.0
                            ),
                            "advantage_status": status,
                        }
                    )
    return output


def aggregate_hmarl_advantage(
    rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    keys = sorted(
        {
            (
                str(row["baseline_source"]),
                str(row["metric"]),
                str(row["degraded_scenario"]),
                str(row["link_group"]),
                float(row["bandwidth_factor"]),
            )
            for row in rows
        }
    )
    for baseline, metric, degraded_scenario, link_group, factor in keys:
        selected = [
            row
            for row in rows
            if row["baseline_source"] == baseline
            and row["metric"] == metric
            and row["degraded_scenario"] == degraded_scenario
        ]
        statuses = [str(row["advantage_status"]) for row in selected]
        advantage_change_mean = safe_mean(
            [
                float(row["advantage_change_degraded_minus_nominal"])
                for row in selected
            ]
        )
        if statuses and all(status == "expanded" for status in statuses):
            aggregate_status = "consistently_expanded"
        elif statuses and all(status == "shrunk" for status in statuses):
            aggregate_status = "consistently_shrunk"
        elif any(status == "vanished_or_reversed" for status in statuses):
            aggregate_status = "vanished_or_reversed_in_some_episodes"
        elif advantage_change_mean > 1e-9:
            aggregate_status = "expanded_on_average"
        elif advantage_change_mean < -1e-9:
            aggregate_status = "shrunk_on_average"
        else:
            aggregate_status = "stable_or_mixed"
        output.append(
            {
                "baseline_source": baseline,
                "baseline_label": SOURCE_LABELS[baseline],
                "metric": metric,
                "degraded_scenario": degraded_scenario,
                "link_group": link_group,
                "bandwidth_factor": factor,
                "paired_episodes": len(selected),
                "nominal_hmarl_advantage_mean": safe_mean(
                    [float(row["nominal_hmarl_advantage"]) for row in selected]
                ),
                "degraded_hmarl_advantage_mean": safe_mean(
                    [float(row["degraded_hmarl_advantage"]) for row in selected]
                ),
                "advantage_change_mean": advantage_change_mean,
                "expanded_count": statuses.count("expanded"),
                "shrunk_count": statuses.count("shrunk"),
                "stable_count": statuses.count("stable"),
                "emerged_count": statuses.count("emerged"),
                "vanished_or_reversed_count": statuses.count(
                    "vanished_or_reversed"
                ),
                "positive_advantage_degraded_fraction": safe_mean(
                    [
                        bool(row["advantage_present_degraded"])
                        for row in selected
                    ]
                ),
                "aggregate_advantage_status": aggregate_status,
            }
        )
    return output


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
    scenarios = scenario_specs(args)
    degraded_scenarios = [
        scenario for scenario, link_group, _ in scenarios if link_group != "none"
    ]

    episode_rows: List[Dict[str, object]] = []
    dag_rows: List[Dict[str, object]] = []
    node_rows_all: List[Dict[str, object]] = []
    workflow_rows_all: List[Dict[str, object]] = []
    task_rows_all: List[Dict[str, object]] = []
    reference_by_episode: Dict[
        Tuple[int, int],
        Tuple[
            frozenset[Tuple[str, str]],
            Dict[str, float],
            Dict[str, tuple],
        ],
    ] = {}

    for source in args.sources:
        checkpoint = resolve_checkpoint(
            templates[source], args.checkpoint_seed
        )
        for seed in args.seeds:
            for episode in range(args.episodes_per_seed):
                for scenario, link_group, factor in scenarios:
                    print(
                        f"Replaying {SOURCE_LABELS[source]}: scenario={scenario}, "
                        f"seed={seed}, episode={episode + 1}/"
                        f"{args.episodes_per_seed}"
                    )
                    (
                        summary,
                        task_rows,
                        workflow_rows,
                        node_rows,
                        edges,
                        deadlines,
                        audit,
                        workload_signature,
                    ) = replay_episode(
                        args=args,
                        source=source,
                        scenario=scenario,
                        link_group=link_group,
                        bandwidth_factor=factor,
                        base_seed=int(seed),
                        episode=episode,
                        checkpoint=checkpoint,
                        device=device,
                    )
                    key = (int(summary["episode_seed"]), int(summary["episode"]))
                    reference = reference_by_episode.get(key)
                    if reference is None:
                        reference_by_episode[key] = (
                            edges,
                            deadlines,
                            workload_signature,
                        )
                    else:
                        (
                            reference_edges,
                            reference_deadlines,
                            reference_workload_signature,
                        ) = reference
                        if edges != reference_edges:
                            raise RuntimeError(
                                f"Global DAG differs across schedulers/scenarios for {key}."
                            )
                        if deadlines.keys() != reference_deadlines.keys() or any(
                            not np.isclose(
                                deadlines[item],
                                reference_deadlines[item],
                                rtol=1e-10,
                                atol=1e-10,
                            )
                            for item in deadlines
                        ):
                            raise RuntimeError(
                                f"Workflow deadlines differ across paired runs for {key}."
                            )
                        if workload_signature != reference_workload_signature:
                            raise RuntimeError(
                                f"Task attributes or privacy/resource admissible sets "
                                f"differ across paired runs for {key}."
                            )

                    episode_rows.append(summary)
                    common = {
                        "source": source,
                        "source_label": SOURCE_LABELS[source],
                        "scenario": scenario,
                        "link_group": link_group,
                        "bandwidth_factor": factor,
                        "base_seed": int(seed),
                        "episode": episode + 1,
                        "episode_seed": int(summary["episode_seed"]),
                        "horizon_s": float(summary["horizon_s"]),
                        "action_mask_protocol": args.action_mask_protocol,
                    }
                    dag_rows.append(
                        {
                            **common,
                            **audit,
                            "global_dag_matches_reference": True,
                            "deadlines_match_reference": True,
                            "task_attributes_match_reference": True,
                            "privacy_resource_masks_match_reference": True,
                        }
                    )
                    node_rows_all.extend({**common, **row} for row in node_rows)
                    workflow_rows_all.extend(
                        {**common, **row} for row in workflow_rows
                    )
                    task_rows_all.extend({**common, **row} for row in task_rows)

    add_scenario_ranks(episode_rows)
    source_summary = aggregate_source_scenarios(episode_rows)
    degradation_rows = paired_degradation_rows(
        episode_rows, degraded_scenarios
    )
    advantage_rows = hmarl_advantage_rows(episode_rows, degraded_scenarios)
    advantage_summary = aggregate_hmarl_advantage(advantage_rows)
    adaptation_rows = baseline_adaptation_rows(args)
    scenario_count = len(scenarios)
    degradation_count = scenario_count - 1
    requested_episode_rows = (
        len(args.sources)
        * len(args.seeds)
        * args.episodes_per_seed
        * scenario_count
    )
    requested_degradation_rows = (
        len(args.sources)
        * len(args.seeds)
        * args.episodes_per_seed
        * degradation_count
    )
    requested_advantage_rows = (
        (len(args.sources) - 1 if "h_marl" in args.sources else 0)
        * len(args.seeds)
        * args.episodes_per_seed
        * 4
        * degradation_count
    )
    complete_requested_sweep = (
        args.link_groups == ["pi_cloud", "pi_edge", "edge_cloud"]
        and args.bandwidth_factors == [0.90, 0.70, 0.50]
    )
    canonical_artifacts = (
        args.predictor == DEFAULT_PREDICTOR.expanduser().resolve()
        and args.hmarl_checkpoint_template
        == DEFAULT_CHECKPOINT_TEMPLATES["h_marl"]
        and args.hyperjet_checkpoint_template
        == DEFAULT_CHECKPOINT_TEMPLATES["hyperjet"]
        and args.maddpg_checkpoint_template
        == DEFAULT_CHECKPOINT_TEMPLATES["maddpg_ra"]
        and args.a2c_checkpoint_template
        == DEFAULT_CHECKPOINT_TEMPLATES["a2c_drl"]
    )
    requested_run_complete = (
        len(episode_rows) == requested_episode_rows
        and len(degradation_rows) == requested_degradation_rows
        and len(advantage_rows) == requested_advantage_rows
    )
    publication_ready = (
        args.max_decisions is None
        and args.episodes_per_seed == 1
        and args.load == 100
        and args.templates == [4]
        and args.deadline_factor == 1.10
        and args.action_mask_protocol == "native_system"
        and set(args.sources) == set(SOURCE_LABELS)
        and len(args.sources) == len(SOURCE_LABELS)
        and set(args.seeds) == {42, 52, 62}
        and len(args.seeds) == 3
        and complete_requested_sweep
        and canonical_artifacts
        and len(episode_rows) == 150
        and len(degradation_rows) == 135
        and len(advantage_rows) == 432
        and all(bool(row["publication_ready"]) for row in episode_rows)
    )

    write_csv(args.output_dir / "robustness_episode_metrics.csv", episode_rows)
    write_csv(
        args.output_dir / "robustness_source_scenario_summary.csv",
        source_summary,
    )
    write_csv(
        args.output_dir / "robustness_paired_degradation.csv",
        degradation_rows,
    )
    write_csv(
        args.output_dir / "robustness_hmarl_advantage.csv", advantage_rows
    )
    write_csv(
        args.output_dir / "robustness_hmarl_advantage_summary.csv",
        advantage_summary,
    )
    write_csv(args.output_dir / "global_dag_audit.csv", dag_rows)
    write_csv(
        args.output_dir / "baseline_global_dag_adaptation.csv",
        adaptation_rows,
    )
    write_csv(
        args.output_dir / "robustness_node_utilization.csv", node_rows_all
    )
    write_csv(
        args.output_dir / "robustness_workflow_metrics.csv", workflow_rows_all
    )
    if args.write_task_csv:
        write_csv(
            args.output_dir / "robustness_task_metrics.csv", task_rows_all
        )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reviewer_issue": 9,
        "protocol": (
            "paired deterministic frozen-policy replay on an exactly identical "
            "global DAG; no training or parameter updates"
        ),
        "comparison_scope": (
            "Common-global-DAG deployment replay of the paper's existing frozen "
            "checkpoints. The learned legacy baselines are state-adapted for "
            "evaluation but were not retrained under H-MARL's template/deadline protocol."
        ),
        "degradation": (
            "independently multiply all finite directed links in each selected "
            "cross-tier link group by each selected bandwidth factor after reset; "
            "keep nominal policy-independent deadlines"
        ),
        "sources": args.sources,
        "source_labels": {
            source: SOURCE_LABELS[source] for source in args.sources
        },
        "seeds": args.seeds,
        "checkpoint_seed": args.checkpoint_seed,
        "episodes_per_seed": args.episodes_per_seed,
        "load": args.load,
        "templates": args.templates,
        "deadline_factor": args.deadline_factor,
        "bandwidth_factors": args.bandwidth_factors,
        "link_groups": args.link_groups,
        "scenarios": [
            {
                "scenario": scenario,
                "link_group": link_group,
                "bandwidth_factor": factor,
            }
            for scenario, link_group, factor in scenarios
        ],
        "action_mask_protocol": args.action_mask_protocol,
        "resolved_device": str(device),
        "predictor": str(args.predictor),
        "checkpoint_templates": templates,
        "max_decisions": args.max_decisions,
        "requested_run_complete": requested_run_complete,
        "publication_ready": publication_ready,
        "publication_ready_requirements": {
            "five_unique_schedulers": set(args.sources) == set(SOURCE_LABELS)
            and len(args.sources) == len(SOURCE_LABELS),
            "evaluation_seeds_42_52_62": set(args.seeds) == {42, 52, 62}
            and len(args.seeds) == 3,
            "one_episode_per_seed": args.episodes_per_seed == 1,
            "load_100": args.load == 100,
            "held_out_template_4_only": args.templates == [4],
            "deadline_factor_1_10": args.deadline_factor == 1.10,
            "native_system_action_mask": (
                args.action_mask_protocol == "native_system"
            ),
            "complete_90_70_50_three_link_group_sweep": complete_requested_sweep,
            "canonical_predictor_and_checkpoint_templates": canonical_artifacts,
            "exact_150_episode_rows": len(episode_rows) == 150,
            "exact_135_degradation_pairs": len(degradation_rows) == 135,
            "exact_432_hmarl_advantage_rows": len(advantage_rows) == 432,
            "complete_nominal_degraded_pairs": len(episode_rows)
            == requested_episode_rows
            and len(degradation_rows) == requested_degradation_rows,
            "complete_hmarl_baseline_advantage_pairs": len(advantage_rows)
            == requested_advantage_rows,
            "untruncated": args.max_decisions is None,
        },
        "global_dag_fairness": {
            "same_environment": True,
            "same_task_and_workflow_instances": True,
            "same_global_parent_child_edges": True,
            "same_generated_cross_workflow_edges": True,
            "same_privacy_and_resource_masks": True,
            "same_policy_independent_deadlines": True,
            "same_policy_action_masks": (
                args.action_mask_protocol == "common_basic_mask"
            ),
            "native_system_note": (
                "Under native_system, H-MARL retains its method-specific "
                "predictor-conditioned MILP QoS pruning; every scheduler still "
                "shares the same basic privacy/resource admissible set."
            ),
        },
        "baseline_state_adapter": (
            "HyperJet, MADDPG-RA, and A2C-DRL receive their legacy privacy/3 "
            "feature and static deadline slack before the existing field reorder."
        ),
        "predictor_robustness_scope": (
            "The predictor remains frozen. The default static predictor is "
            "bandwidth-blind; degraded bandwidth acts through execution-time "
            "communication estimates and native MILP pruning, not predictor retraining."
        ),
        "metric_directions": {
            metric: ("lower is better" if lower else "higher is better")
            for metric, lower in LOWER_IS_BETTER.items()
        },
        "per_episode": episode_rows,
        "source_scenario_summary": source_summary,
        "paired_degradation": degradation_rows,
        "hmarl_advantage": advantage_rows,
        "hmarl_advantage_summary": advantage_summary,
        "baseline_global_dag_adaptation": adaptation_rows,
        "argv": sys.argv,
    }
    (args.output_dir / "robustness_summary.json").write_text(
        json.dumps(json_safe(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nGlobal-DAG robustness diagnostics completed.")
    print(f"Output directory: {args.output_dir}")
    print(json.dumps(json_safe(source_summary), ensure_ascii=False, indent=2))
    if args.require_publication_ready and not publication_ready:
        raise RuntimeError(
            "Run completed but did not satisfy the locked publication protocol; "
            "inspect robustness_summary.json publication_ready_requirements."
        )


if __name__ == "__main__":
    main()
