from __future__ import annotations

"""Evaluate MILP-policy coupling with a frozen H-MARL checkpoint.

This script never trains or updates the scheduler.  It deterministically replays
the frozen policy and records, for every decision:

* deadline-relaxation status;
* candidate counts before and after MILP filtering;
* the MILP-preferred node;
* the policy-proposed and actually executed nodes; and
* proposed/executed disagreement with the MILP-preferred node;
* MILP solve time, mask-generation time, and frozen-policy inference time.

The default configuration reproduces the held-out template/seeds used by the
cross-scheduler predictor experiment.  Pass a different checkpoint or seed list
when the paper's primary scheduler evaluation uses another frozen model.
"""

import argparse
import csv
import json
import sys
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_LOGIC_ROOT = SCRIPT_DIR.parents[1]
METHOD_PARENT = TRAINING_LOGIC_ROOT / "03_methods"
if str(METHOD_PARENT) not in sys.path:
    sys.path.insert(0, str(METHOD_PARENT))

from paper_method_hmarl.config import ExperimentConfig, MethodSwitches  # noqa: E402
from paper_method_hmarl.trainer import (  # noqa: E402
    PolicyRuntime,
    build_environment,
    load_predictor,
    resolve_device,
    safe_torch_load,
    set_random_seed,
)


DEFAULT_CHECKPOINT = (
    TRAINING_LOGIC_ROOT
    / "06_results_archive"
    / "E6_paper_method_ablation"
    / "runs"
    / "ablation_full_100users_seed42"
    / "best_model.pt"
)
DEFAULT_PREDICTOR = (
    TRAINING_LOGIC_ROOT
    / "06_results_archive"
    / "E6_paper_method_ablation"
    / "shared"
    / "slack_histgbrt_remaining_v2.joblib"
)
DEFAULT_OUTPUT_ROOT = (
    TRAINING_LOGIC_ROOT
    / "06_results_archive"
    / "E6_paper_method_ablation"
    / "milp_policy_diagnostics"
)


DECISION_FIELDS = (
    "seed",
    "template_ids",
    "deadline_factor",
    "decision_index",
    "task_id",
    "decision_time",
    "milp_preferred_node",
    "policy_proposed_node",
    "executed_node",
    "proposed_disagreement",
    "executed_disagreement",
    "critical_task",
    "deadline_relaxed",
    "slack_qos_active",
    "predicted_critical_slack",
    "basic_candidate_count",
    "returned_candidate_count",
    "pruned_candidate_count",
    "pruning_rate",
    "policy_action_count",
    "relaxed_candidate_count",
    "solver_status",
    "mask_generation_time_ms",
    "milp_solve_time_ms",
    "milp_pruner_time_ms",
    "policy_inference_time_ms",
    "online_decision_time_ms",
    "invalid_action",
    "action_corrected",
)


SEED_SUMMARY_FIELDS = (
    "seed",
    "template_ids",
    "deadline_factor",
    "decisions",
    "truncated",
    "critical_task_count",
    "critical_task_rate",
    "deadline_relaxation_count",
    "deadline_relaxation_rate",
    "critical_relaxation_rate",
    "avg_basic_candidates",
    "avg_returned_candidates",
    "avg_pruned_candidates",
    "avg_pruning_rate",
    "avg_relaxed_basic_candidates",
    "avg_relaxed_returned_candidates",
    "avg_relaxed_pruning_rate",
    "proposed_disagreement_count",
    "proposed_disagreement_rate",
    "executed_disagreement_count",
    "executed_disagreement_rate",
    "relaxed_executed_disagreement_count",
    "relaxed_executed_disagreement_rate",
    "nonrelaxed_executed_disagreement_count",
    "nonrelaxed_executed_disagreement_rate",
    "missing_preferred_node_count",
    "avg_mask_generation_time_ms",
    "avg_milp_solve_time_ms",
    "p95_milp_solve_time_ms",
    "avg_milp_pruner_time_ms",
    "avg_policy_inference_time_ms",
    "avg_online_decision_time_ms",
    "p95_online_decision_time_ms",
    "avg_wf_time",
    "ddl_rate",
    "avg_critical_slack",
)


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Replay a frozen H-MARL policy and report MILP relaxation, pruning, "
            "and policy-preferred-node disagreement. No training is performed."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--predictor",
        type=Path,
        default=DEFAULT_PREDICTOR,
        help="Slack predictor used by the frozen scheduler during replay.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[92, 102, 112, 122],
        help="Independent held-out environment seeds.",
    )
    parser.add_argument(
        "--templates",
        type=int,
        nargs="+",
        default=[4],
        help="Held-out workflow template IDs.",
    )
    parser.add_argument(
        "--deadline-factors",
        type=float,
        nargs="+",
        default=[1.10, 1.25, 1.50],
        help="Deadline factors evaluated with the same frozen policy.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional debugging cap. Omit for the full 10,400-decision episode.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress after this many decisions; use 0 to disable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / f"run_{timestamp}",
    )
    parser.add_argument(
        "--no-decision-csv",
        action="store_true",
        help="Write only seed and aggregate summaries.",
    )
    return parser.parse_args()


def load_experiment_config(
    checkpoint_payload: Mapping[str, object],
    *,
    predictor_path: Path,
    templates: Sequence[int],
    device: str,
) -> ExperimentConfig:
    raw_config = checkpoint_payload.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError(
            "The checkpoint does not contain its ExperimentConfig. "
            "Use a paper_method_hmarl checkpoint with a saved 'config' entry."
        )

    allowed = {field.name for field in fields(ExperimentConfig)}
    payload = {key: value for key, value in raw_config.items() if key in allowed}
    raw_switches = payload.get("switches", {})
    if isinstance(raw_switches, MethodSwitches):
        switches = raw_switches
    elif isinstance(raw_switches, Mapping):
        switches = MethodSwitches(**raw_switches)
    else:
        raise TypeError(f"Unsupported checkpoint switches: {type(raw_switches)!r}")

    payload["switches"] = switches
    payload["test_templates"] = tuple(sorted(set(int(value) for value in templates)))
    payload["max_steps_per_episode"] = None
    payload["device"] = device
    if switches.slack_predictor:
        payload["slack_model_path"] = str(predictor_path.resolve())

    config = ExperimentConfig(**payload)
    if not config.switches.milp_pruning:
        raise ValueError("Problem-4 diagnostics require a checkpoint with MILP pruning enabled.")
    if not config.switches.action_mask:
        print(
            "WARNING: action_mask is disabled in this checkpoint. The policy may select "
            "nodes outside the MILP-returned set; disagreement remains measurable."
        )
    return config


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def normalized_node(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def initialize_totals() -> MutableMapping[str, object]:
    return {
        "decisions": 0.0,
        "critical_tasks": 0.0,
        "deadline_relaxations": 0.0,
        "basic_candidates": 0.0,
        "returned_candidates": 0.0,
        "pruned_candidates": 0.0,
        "pruning_rate_sum": 0.0,
        "proposed_disagreements": 0.0,
        "executed_disagreements": 0.0,
        "relaxed_executed_disagreements": 0.0,
        "nonrelaxed_executed_disagreements": 0.0,
        "missing_preferred_nodes": 0.0,
        "relaxed_basic_candidates": 0.0,
        "relaxed_returned_candidates": 0.0,
        "relaxed_pruning_rate_sum": 0.0,
        "mask_generation_time_ms": 0.0,
        "milp_solve_time_ms": 0.0,
        "milp_pruner_time_ms": 0.0,
        "policy_inference_time_ms": 0.0,
        "online_decision_time_ms": 0.0,
        "milp_solve_time_samples_ms": [],
        "online_decision_time_samples_ms": [],
    }


def update_totals(totals: MutableMapping[str, object], row: Mapping[str, object]) -> None:
    relaxed = bool(row["deadline_relaxed"])
    totals["decisions"] += 1.0
    totals["critical_tasks"] += float(bool(row["critical_task"]))
    totals["deadline_relaxations"] += float(relaxed)
    totals["basic_candidates"] += float(row["basic_candidate_count"])
    totals["returned_candidates"] += float(row["returned_candidate_count"])
    totals["pruned_candidates"] += float(row["pruned_candidate_count"])
    totals["pruning_rate_sum"] += float(row["pruning_rate"])
    for metric in (
        "mask_generation_time_ms",
        "milp_solve_time_ms",
        "milp_pruner_time_ms",
        "policy_inference_time_ms",
        "online_decision_time_ms",
    ):
        totals[metric] += float(row[metric])
    totals["milp_solve_time_samples_ms"].append(float(row["milp_solve_time_ms"]))
    totals["online_decision_time_samples_ms"].append(
        float(row["online_decision_time_ms"])
    )

    if not row["milp_preferred_node"]:
        totals["missing_preferred_nodes"] += 1.0
        return

    proposed_disagreement = bool(row["proposed_disagreement"])
    executed_disagreement = bool(row["executed_disagreement"])
    totals["proposed_disagreements"] += float(proposed_disagreement)
    totals["executed_disagreements"] += float(executed_disagreement)
    if relaxed:
        totals["relaxed_executed_disagreements"] += float(executed_disagreement)
        totals["relaxed_basic_candidates"] += float(row["basic_candidate_count"])
        totals["relaxed_returned_candidates"] += float(
            row["returned_candidate_count"]
        )
        totals["relaxed_pruning_rate_sum"] += float(row["pruning_rate"])
    else:
        totals["nonrelaxed_executed_disagreements"] += float(executed_disagreement)


def totals_to_summary(
    *,
    seed: int | str,
    templates: Sequence[int],
    deadline_factor: float,
    totals: Mapping[str, object],
    truncated: bool,
    final_info: Mapping[str, object] | None,
) -> Dict[str, object]:
    decisions = float(totals["decisions"])
    relaxed = float(totals["deadline_relaxations"])
    nonrelaxed = decisions - relaxed
    info = final_info or {}
    milp_samples = np.asarray(totals["milp_solve_time_samples_ms"], dtype=np.float64)
    online_samples = np.asarray(
        totals["online_decision_time_samples_ms"], dtype=np.float64
    )
    return {
        "seed": seed,
        "template_ids": ";".join(str(value) for value in templates),
        "deadline_factor": float(deadline_factor),
        "decisions": int(decisions),
        "truncated": bool(truncated),
        "critical_task_count": int(totals["critical_tasks"]),
        "critical_task_rate": safe_divide(totals["critical_tasks"], decisions),
        "deadline_relaxation_count": int(relaxed),
        "deadline_relaxation_rate": safe_divide(relaxed, decisions),
        "critical_relaxation_rate": safe_divide(relaxed, totals["critical_tasks"]),
        "avg_basic_candidates": safe_divide(totals["basic_candidates"], decisions),
        "avg_returned_candidates": safe_divide(
            totals["returned_candidates"], decisions
        ),
        "avg_pruned_candidates": safe_divide(totals["pruned_candidates"], decisions),
        "avg_pruning_rate": safe_divide(totals["pruning_rate_sum"], decisions),
        "avg_relaxed_basic_candidates": safe_divide(
            totals["relaxed_basic_candidates"], relaxed
        ),
        "avg_relaxed_returned_candidates": safe_divide(
            totals["relaxed_returned_candidates"], relaxed
        ),
        "avg_relaxed_pruning_rate": safe_divide(
            totals["relaxed_pruning_rate_sum"], relaxed
        ),
        "proposed_disagreement_count": int(totals["proposed_disagreements"]),
        "proposed_disagreement_rate": safe_divide(
            totals["proposed_disagreements"], decisions
        ),
        "executed_disagreement_count": int(totals["executed_disagreements"]),
        "executed_disagreement_rate": safe_divide(
            totals["executed_disagreements"], decisions
        ),
        "relaxed_executed_disagreement_count": int(
            totals["relaxed_executed_disagreements"]
        ),
        "relaxed_executed_disagreement_rate": safe_divide(
            totals["relaxed_executed_disagreements"], relaxed
        ),
        "nonrelaxed_executed_disagreement_count": int(
            totals["nonrelaxed_executed_disagreements"]
        ),
        "nonrelaxed_executed_disagreement_rate": safe_divide(
            totals["nonrelaxed_executed_disagreements"], nonrelaxed
        ),
        "missing_preferred_node_count": int(totals["missing_preferred_nodes"]),
        "avg_mask_generation_time_ms": safe_divide(
            float(totals["mask_generation_time_ms"]), decisions
        ),
        "avg_milp_solve_time_ms": safe_divide(
            float(totals["milp_solve_time_ms"]), decisions
        ),
        "p95_milp_solve_time_ms": float(np.percentile(milp_samples, 95))
        if milp_samples.size
        else 0.0,
        "avg_milp_pruner_time_ms": safe_divide(
            float(totals["milp_pruner_time_ms"]), decisions
        ),
        "avg_policy_inference_time_ms": safe_divide(
            float(totals["policy_inference_time_ms"]), decisions
        ),
        "avg_online_decision_time_ms": safe_divide(
            float(totals["online_decision_time_ms"]), decisions
        ),
        "p95_online_decision_time_ms": float(np.percentile(online_samples, 95))
        if online_samples.size
        else 0.0,
        "avg_wf_time": float(info.get("avg_wf_time", 0.0)),
        "ddl_rate": float(info.get("ddl_rate", 0.0)),
        "avg_critical_slack": float(info.get("avg_critical_slack", 0.0)),
    }


def merge_totals(
    target: MutableMapping[str, object], source: Mapping[str, object]
) -> None:
    for key in target:
        if isinstance(target[key], list):
            target[key].extend(source[key])
        else:
            target[key] += float(source[key])


def evaluate_seed(
    *,
    seed: int,
    config: ExperimentConfig,
    predictor,
    runtime: PolicyRuntime,
    decision_writer: csv.DictWriter | None,
    templates: Sequence[int],
    max_steps: int | None,
    progress_every: int,
) -> tuple[Dict[str, object], MutableMapping[str, object]]:
    set_random_seed(seed)
    env = build_environment(
        config,
        predictor,
        seed_offset=seed - int(config.seed),
        evaluation=True,
    )
    state = env.reset()
    masks = env._get_action_mask()
    totals = initialize_totals()
    done = False
    step = 0
    final_info: Dict[str, object] = {}

    while not done:
        task = env.current_task
        decision_time = float(getattr(env, "current_time", 0.0))
        diagnostics = dict(env.last_mask_diagnostics)
        preferred_node = normalized_node(diagnostics.get("selected_node"))
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=runtime.device
        ).unsqueeze(0)
        policy_started_at = perf_counter()
        with torch.no_grad():
            tier_action, node_action, _ = runtime.select_action(
                state_tensor, masks, deterministic=True
            )
        policy_inference_time_ms = (perf_counter() - policy_started_at) * 1000.0
        next_state, _, done, next_masks, info = env.step(tier_action, node_action)

        proposed_node = normalized_node(info.get("proposed_node"))
        executed_node = normalized_node(info.get("executed_node"))
        has_preferred = bool(preferred_node)
        row: Dict[str, object] = {
            "seed": seed,
            "template_ids": ";".join(str(value) for value in templates),
            "deadline_factor": float(config.deadline_factor),
            "decision_index": step,
            "task_id": getattr(task, "unique_id", ""),
            "decision_time": decision_time,
            "milp_preferred_node": preferred_node,
            "policy_proposed_node": proposed_node,
            "executed_node": executed_node,
            "proposed_disagreement": bool(
                has_preferred and proposed_node != preferred_node
            ),
            "executed_disagreement": bool(
                has_preferred and executed_node != preferred_node
            ),
            "critical_task": bool(diagnostics.get("critical_task", False)),
            "deadline_relaxed": bool(diagnostics.get("deadline_relaxed", False)),
            "slack_qos_active": bool(diagnostics.get("slack_qos_active", False)),
            "predicted_critical_slack": float(
                diagnostics.get("predicted_critical_slack", 0.0)
            ),
            "basic_candidate_count": int(
                diagnostics.get("current_basic_candidates", 0)
            ),
            "returned_candidate_count": int(
                diagnostics.get("current_feasible_actions", 0)
            ),
            "pruned_candidate_count": int(
                diagnostics.get("current_pruned_actions", 0)
            ),
            "pruning_rate": float(diagnostics.get("current_pruning_rate", 0.0)),
            "policy_action_count": int(diagnostics.get("policy_action_count", 0)),
            "relaxed_candidate_count": int(
                diagnostics.get("relaxed_candidate_count", 0)
            ),
            "solver_status": diagnostics.get("solver_status", ""),
            "mask_generation_time_ms": float(
                diagnostics.get("mask_generation_time_ms", 0.0)
            ),
            "milp_solve_time_ms": float(
                diagnostics.get("milp_solve_time_ms", 0.0)
            ),
            "milp_pruner_time_ms": float(
                diagnostics.get("milp_pruner_time_ms", 0.0)
            ),
            "policy_inference_time_ms": float(policy_inference_time_ms),
            "online_decision_time_ms": float(
                diagnostics.get("mask_generation_time_ms", 0.0)
                + policy_inference_time_ms
            ),
            "invalid_action": bool(info.get("invalid_action", False)),
            "action_corrected": bool(info.get("action_corrected", False)),
        }
        update_totals(totals, row)
        if decision_writer is not None:
            decision_writer.writerow(row)

        final_info = dict(info)
        state, masks = next_state, next_masks
        step += 1
        if progress_every and step % progress_every == 0:
            print(
                f"seed={seed} decisions={step} "
                f"relax={safe_divide(totals['deadline_relaxations'], step):.4f} "
                f"executed_disagree={safe_divide(totals['executed_disagreements'], step):.4f}"
            )
        if max_steps is not None and step >= max_steps:
            break

    truncated = not done
    summary = totals_to_summary(
        seed=seed,
        templates=templates,
        deadline_factor=config.deadline_factor,
        totals=totals,
        truncated=truncated,
        final_info=final_info,
    )
    return summary, totals


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields_: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields_))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.predictor = args.predictor.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.predictor.exists():
        raise FileNotFoundError(f"Predictor not found: {args.predictor}")
    if not args.seeds:
        raise ValueError("At least one evaluation seed is required.")
    if any(template not in range(5) for template in args.templates):
        raise ValueError(f"Template IDs must be in [0, 4], got {args.templates}")
    if any(factor < 1.0 for factor in args.deadline_factors):
        raise ValueError(
            f"All --deadline-factors must be at least 1.0, got {args.deadline_factors}"
        )
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive when provided.")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = resolve_device(args.device)
    checkpoint_payload = safe_torch_load(args.checkpoint, device)
    config = load_experiment_config(
        checkpoint_payload,
        predictor_path=args.predictor,
        templates=args.templates,
        device=args.device,
    )
    predictor = load_predictor(config)

    probe_env = build_environment(
        config,
        predictor,
        seed_offset=int(args.seeds[0]) - int(config.seed),
        evaluation=True,
    )
    initial_state = probe_env.reset()
    runtime = PolicyRuntime(config, len(initial_state), probe_env.tier_sizes, device)
    if "runtime" not in checkpoint_payload:
        raise ValueError(f"Checkpoint has no H-MARL runtime: {args.checkpoint}")
    runtime.load_state_dict(checkpoint_payload["runtime"], load_optimizers=False)
    for module in runtime.policies.values():
        module.eval()
    runtime.critic.eval()

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "frozen deterministic replay; no optimizer or model update",
        "checkpoint": str(args.checkpoint),
        "predictor": str(args.predictor),
        "device": str(device),
        "seeds": [int(value) for value in args.seeds],
        "templates": [int(value) for value in args.templates],
        "deadline_factors": [float(value) for value in args.deadline_factors],
        "max_steps": args.max_steps,
        "config_seed": int(config.seed),
        "num_users": int(config.num_users),
        "implementation_version": config.implementation_version,
        "switches": {
            "hierarchical": config.switches.hierarchical,
            "slack_predictor": config.switches.slack_predictor,
            "milp_pruning": config.switches.milp_pruning,
            "action_mask": config.switches.action_mask,
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    decision_handle = None
    decision_writer = None
    if not args.no_decision_csv:
        decision_handle = (args.output_dir / "milp_policy_decisions.csv").open(
            "w", encoding="utf-8-sig", newline=""
        )
        decision_writer = csv.DictWriter(
            decision_handle, fieldnames=list(DECISION_FIELDS)
        )
        decision_writer.writeheader()

    seed_rows: List[Dict[str, object]] = []
    factor_rows: List[Dict[str, object]] = []
    try:
        for deadline_factor in args.deadline_factors:
            factor_config = replace(config, deadline_factor=float(deadline_factor))
            factor_totals = initialize_totals()
            factor_seed_rows: List[Dict[str, object]] = []
            print(
                f"\n=== Deadline factor {deadline_factor:.2f}: "
                f"{len(args.seeds)} frozen-policy seeds ==="
            )
            for seed in args.seeds:
                print(
                    f"Evaluating frozen policy: factor={deadline_factor:.2f}, "
                    f"seed={seed}, templates={tuple(args.templates)}"
                )
                seed_summary, seed_totals = evaluate_seed(
                    seed=int(seed),
                    config=factor_config,
                    predictor=predictor,
                    runtime=runtime,
                    decision_writer=decision_writer,
                    templates=args.templates,
                    max_steps=args.max_steps,
                    progress_every=args.progress_every,
                )
                seed_rows.append(seed_summary)
                factor_seed_rows.append(seed_summary)
                merge_totals(factor_totals, seed_totals)
                if decision_handle is not None:
                    decision_handle.flush()
                print(
                    f"Completed factor={deadline_factor:.2f}, seed={seed}: "
                    f"decisions={seed_summary['decisions']}, "
                    f"relax={seed_summary['deadline_relaxation_rate']:.4f}, "
                    f"executed_disagree={seed_summary['executed_disagreement_rate']:.4f}"
                )

            factor_row = totals_to_summary(
                seed="ALL",
                templates=args.templates,
                deadline_factor=float(deadline_factor),
                totals=factor_totals,
                truncated=any(bool(row["truncated"]) for row in factor_seed_rows),
                final_info=None,
            )
            for metric in ("avg_wf_time", "ddl_rate", "avg_critical_slack"):
                factor_row[metric] = float(
                    np.mean([float(row[metric]) for row in factor_seed_rows])
                )
            factor_rows.append(factor_row)
    finally:
        if decision_handle is not None:
            decision_handle.close()

    write_csv(
        args.output_dir / "milp_policy_seed_summary.csv",
        seed_rows,
        SEED_SUMMARY_FIELDS,
    )
    write_csv(
        args.output_dir / "milp_policy_summary.csv",
        factor_rows,
        SEED_SUMMARY_FIELDS,
    )
    summary_payload = {
        "metadata": metadata,
        "per_seed": seed_rows,
        "per_deadline_factor": factor_rows,
        "metric_definitions": {
            "deadline_relaxation_rate": "decisions that restore top-k minimum-tardiness candidates / all decisions",
            "critical_relaxation_rate": "relaxed decisions / critical decisions",
            "proposed_disagreement_rate": "policy-proposed node differs from MILP preferred node / all decisions",
            "executed_disagreement_rate": "actually executed node differs from MILP preferred node / all decisions",
            "relaxed_executed_disagreement_rate": "executed disagreement / relaxed decisions",
            "nonrelaxed_executed_disagreement_rate": "executed disagreement / non-relaxed decisions",
            "avg_pruning_rate": "mean per-decision fraction removed from the privacy/resource-feasible set",
            "avg_relaxed_pruning_rate": "mean candidate pruning fraction conditioned on deadline relaxation",
            "avg_milp_solve_time_ms": "mean SciPy MILP solver wall-clock time per scheduling decision",
            "p95_milp_solve_time_ms": "95th percentile SciPy MILP solver wall-clock time",
            "avg_online_decision_time_ms": "mean causal action-mask generation plus frozen-policy inference time",
            "p95_online_decision_time_ms": "95th percentile causal action-mask generation plus frozen-policy inference time",
        },
    }
    (args.output_dir / "milp_policy_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nFrozen-policy MILP diagnostics completed.")
    print(f"Output directory: {args.output_dir}")
    print(json.dumps(factor_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
