from __future__ import annotations

"""Reviewer issue 10: frozen H-MARL runtime and GPU-capacity diagnostics.

This script never trains or updates the scheduler.  It loads the existing Full
H-MARL checkpoint and deterministically replays held-out template 4 to report:

* predictor inference time per decision;
* MILP solve/pruner and complete online decision latency;
* process and optional CUDA memory;
* whether GPU memory removes otherwise privacy/CPU/RAM-feasible actions.

The nominal trace is evaluated on seeds 42, 52, and 62.  One additional frozen
replay reduces Edge GPU capacity to 10% so the GPU-memory constraint is actually
exercised without changing policy parameters.  The environment is a serial
single-server model: resources are restored at a task's candidate start time,
so it does not model concurrent GPU-memory residency on a busy node.  The output
states this explicitly rather than claiming a dynamic contention event.
"""

import argparse
import csv
import gc
import json
import math
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Mapping, MutableMapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_privacy_ablation import (  # noqa: E402
    DEFAULT_ARCHIVE_RUNS,
    DEFAULT_PREDICTOR,
    TRAINING_LOGIC_ROOT,
    checkpoint_config,
)
from paper_method_hmarl.trainer import (  # noqa: E402
    PolicyRuntime,
    build_environment,
    load_predictor,
    resolve_device,
    safe_torch_load,
    set_random_seed,
)


DEFAULT_CHECKPOINT = (
    DEFAULT_ARCHIVE_RUNS
    / "ablation_full_100users_seed42_deadline_v2"
    / "best_model.pt"
)
DEFAULT_OUTPUT_DIR = (
    TRAINING_LOGIC_ROOT
    / "06_results_archive"
    / "E6_paper_method_ablation"
    / "runtime_gpu_diagnostics"
    / "frozen_full_seed42"
)

TIMING_METRICS = (
    "predictor_inference_time_ms",
    "predictor_state_time_ms",
    "predictor_mask_time_ms",
    "predictor_transition_time_ms",
    "milp_solve_time_ms",
    "milp_pruner_time_ms",
    "mask_generation_time_ms",
    "state_generation_time_ms",
    "state_tensor_time_ms",
    "policy_inference_time_ms",
    "online_decision_time_ms",
    "environment_transition_time_ms",
    "environment_step_total_time_ms",
    "process_rss_sample_mb",
    "cuda_allocated_sample_mb",
    "cuda_reserved_sample_mb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile frozen Full H-MARL runtime and exercise the simulated "
            "GPU-memory constraint without retraining."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--slack-model", type=Path, default=DEFAULT_PREDICTOR)
    parser.add_argument(
        "--evaluation-seeds", type=int, nargs="+", default=[42, 52, 62]
    )
    parser.add_argument("--nominal-episodes", type=int, default=1)
    parser.add_argument("--stress-seed", type=int, default=42)
    parser.add_argument("--stress-episodes", type=int, default=1)
    parser.add_argument(
        "--edge-gpu-stress-factor",
        type=float,
        default=0.10,
        help="Multiplier applied only to Edge-node GPU memory in the stress replay.",
    )
    parser.add_argument("--warmup-decisions", type=int, default=20)
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=None,
        help="Debug-only episode truncation; omit for complete results.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-binding-events", type=int, default=1000)
    parser.add_argument(
        "--write-decision-csv",
        action="store_true",
        help="Write every decision row; disabled by default to limit disk usage.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero unless the locked four-episode protocol completes.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.evaluation_seeds:
        raise ValueError("At least one evaluation seed is required.")
    if len(set(args.evaluation_seeds)) != len(args.evaluation_seeds):
        raise ValueError("--evaluation-seeds must not contain duplicates.")
    if args.nominal_episodes <= 0 or args.stress_episodes <= 0:
        raise ValueError("Episode counts must be positive.")
    if not 0.0 < args.edge_gpu_stress_factor < 1.0:
        raise ValueError("--edge-gpu-stress-factor must be between 0 and 1.")
    if args.warmup_decisions < 0:
        raise ValueError("--warmup-decisions must be non-negative.")
    if args.max_decisions is not None and args.max_decisions <= 0:
        raise ValueError("--max-decisions must be positive when provided.")
    if args.max_binding_events < 0:
        raise ValueError("--max-binding-events must be non-negative.")


def json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def descriptive(values: Sequence[float]) -> Dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def add_metric_stats(
    row: MutableMapping[str, object],
    values: Mapping[str, Sequence[float]],
) -> None:
    for metric in TIMING_METRICS:
        for statistic, value in descriptive(values.get(metric, ())).items():
            row[f"{metric}_{statistic}"] = value


def process_rss_bytes() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError):
        pass

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            )
            return int(counters.WorkingSetSize) if ok else 0
        except (AttributeError, OSError):
            return 0

    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class PredictorProbe:
    def __init__(self, predictor) -> None:
        self.predictor = predictor
        self.total_ms = 0.0
        self.calls = 0
        self.cache_hits = 0
        self.original: Callable = predictor.predict_remaining_time

        def timed_predict(task):
            cache_hit = False
            if getattr(predictor, "uses_environment_context", False):
                key = (
                    getattr(predictor, "_context_generation", 0),
                    str(task.unique_id),
                )
                cache_hit = key in getattr(
                    predictor, "_context_prediction_cache", {}
                )
            else:
                cache_hit = str(task.unique_id) in getattr(
                    predictor, "prediction_cache", {}
                )
            started = perf_counter()
            result = self.original(task)
            self.total_ms += (perf_counter() - started) * 1000.0
            self.calls += 1
            self.cache_hits += int(cache_hit)
            return result

        predictor.predict_remaining_time = timed_predict

    def snapshot(self) -> tuple[float, int, int]:
        return self.total_ms, self.calls, self.cache_hits

    @staticmethod
    def delta(
        before: tuple[float, int, int], after: tuple[float, int, int]
    ) -> Dict[str, float | int]:
        return {
            "predictor_time_ms": max(0.0, after[0] - before[0]),
            "predictor_calls": max(0, after[1] - before[1]),
            "predictor_cache_hits": max(0, after[2] - before[2]),
        }


def timed_environment_call(
    original: Callable,
    predictor_probe: PredictorProbe,
    records: List[Dict[str, object]],
    include_mask_diagnostics: bool,
    env,
) -> Callable:
    def wrapped():
        before = predictor_probe.snapshot()
        started = perf_counter()
        result = original()
        wall_ms = (perf_counter() - started) * 1000.0
        after = predictor_probe.snapshot()
        record: Dict[str, object] = {
            "wall_time_ms": float(wall_ms),
            **PredictorProbe.delta(before, after),
        }
        if include_mask_diagnostics:
            record.update(dict(env.last_mask_diagnostics))
        records.append(record)
        return result

    return wrapped


def apply_edge_gpu_factor(env, factor: float) -> Dict[str, float | int]:
    before = [float(env.node_specs[node]["gpu_mem"]) for node in env.jetsons]
    for node in env.jetsons:
        env.node_specs[node]["gpu_mem"] = float(
            env.node_specs[node]["gpu_mem"]
        ) * factor
    after = [float(env.node_specs[node]["gpu_mem"]) for node in env.jetsons]
    return {
        "affected_edge_nodes": len(env.jetsons),
        "edge_gpu_mem_before_mean": float(np.mean(before)),
        "edge_gpu_mem_after_mean": float(np.mean(after)),
    }


def gpu_binding_snapshot(env, task) -> Dict[str, object]:
    req_cpu = float(task.core_count)
    req_mem = float(task.req_mem)
    req_gpu = float(task.req_gpu_mem)
    gpu_only_blocked: List[str] = []
    non_gpu_candidates = 0
    feasible_with_gpu = 0
    blocked_by_tier = {"pi": 0, "edge": 0, "cloud": 0}
    for node in env.all_nodes:
        specs = env.node_specs[node]
        non_gpu_ok = bool(
            node in task.allowed_nodes
            and float(specs["cpu"]) + 1e-9 >= req_cpu
            and float(specs["mem"]) + 1e-9 >= req_mem
        )
        if not non_gpu_ok:
            continue
        non_gpu_candidates += 1
        if float(specs["gpu_mem"]) + 1e-9 >= req_gpu:
            feasible_with_gpu += 1
        else:
            gpu_only_blocked.append(node)
            blocked_by_tier[str(specs["type"])] += 1
    return {
        "task_gpu_request": req_gpu,
        "task_uses_gpu": req_gpu > 0.0,
        "non_gpu_candidate_count": non_gpu_candidates,
        "gpu_feasible_candidate_count": feasible_with_gpu,
        "gpu_only_blocked_count": len(gpu_only_blocked),
        "gpu_constraint_binding": bool(gpu_only_blocked),
        "gpu_only_blocked_pi": blocked_by_tier["pi"],
        "gpu_only_blocked_edge": blocked_by_tier["edge"],
        "gpu_only_blocked_cloud": blocked_by_tier["cloud"],
        "gpu_only_blocked_nodes": ";".join(gpu_only_blocked),
    }


def run_profiled_episode(
    *,
    args: argparse.Namespace,
    config,
    policy_state: MutableMapping[str, object],
    runtime: PolicyRuntime | None,
    device: torch.device,
    evaluation_seed: int,
    episode: int,
    scenario: str,
    edge_gpu_factor: float,
    scenario_values: MutableMapping[str, List[float]],
    scenario_counts: MutableMapping[str, float],
    binding_events: List[Dict[str, object]],
    decision_writer: csv.DictWriter | None,
) -> tuple[PolicyRuntime, Dict[str, object]]:
    set_random_seed(evaluation_seed + episode)
    predictor = load_predictor(config)
    env = build_environment(
        config,
        predictor,
        seed_offset=evaluation_seed - int(config.seed),
        evaluation=True,
    )
    capacity_audit = apply_edge_gpu_factor(env, edge_gpu_factor)
    env.reset()

    if runtime is None:
        initial_state = env._get_state()
        runtime = PolicyRuntime(config, len(initial_state), env.tier_sizes, device)
        if not policy_state:
            raise ValueError("Checkpoint has no actor policy state payload.")
        if set(policy_state) != set(runtime.policies):
            raise ValueError(
                "Checkpoint actor modules do not match the evaluation runtime: "
                f"checkpoint={sorted(policy_state)}, "
                f"runtime={sorted(runtime.policies)}."
            )
        for name, state in policy_state.items():
            if name not in runtime.policies:
                raise ValueError(f"Unexpected policy module in checkpoint: {name!r}.")
            runtime.policies[name].load_state_dict(state)
        for module in runtime.policies.values():
            module.eval()
        # select_action() uses actors only. Move the training-only critic off
        # CUDA before the baseline so CUDA memory is the actor inference
        # footprint rather than the complete training runtime.
        runtime.critic.to(torch.device("cpu"))
        runtime.critic.eval()
        policy_state.clear()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    predictor_probe = PredictorProbe(predictor)
    state_records: List[Dict[str, object]] = []
    mask_records: List[Dict[str, object]] = []
    env._get_state = timed_environment_call(
        env._get_state, predictor_probe, state_records, False, env
    )
    env._get_action_mask = timed_environment_call(
        env._get_action_mask, predictor_probe, mask_records, True, env
    )

    if predictor is not None:
        predictor.invalidate_context_cache()
    state = env._get_state()
    masks = env._get_action_mask()

    if device.type == "cuda":
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        cuda_baseline_allocated = int(torch.cuda.memory_allocated(device))
        cuda_baseline_reserved = int(torch.cuda.memory_reserved(device))
    else:
        cuda_baseline_allocated = 0
        cuda_baseline_reserved = 0

    rss_baseline = process_rss_bytes()
    rss_peak = rss_baseline
    values: Dict[str, List[float]] = defaultdict(list)
    decisions = 0
    timing_decisions = 0
    gpu_task_decisions = 0
    gpu_binding_decisions = 0
    gpu_only_blocked_actions = 0
    final_info: Dict[str, object] = {}
    done = False

    while not done:
        if env.current_task is None:
            break
        task = env.current_task
        current_state_record = state_records[-1]
        current_mask_record = mask_records[-1]
        binding = gpu_binding_snapshot(env, task)
        gpu_task_decisions += int(bool(binding["task_uses_gpu"]))
        gpu_binding_decisions += int(bool(binding["gpu_constraint_binding"]))
        gpu_only_blocked_actions += int(binding["gpu_only_blocked_count"])

        synchronize(device)
        tensor_started = perf_counter()
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)
        synchronize(device)
        state_tensor_time_ms = (perf_counter() - tensor_started) * 1000.0

        synchronize(device)
        policy_started = perf_counter()
        with torch.no_grad():
            tier, node, _ = runtime.select_action(
                state_tensor, masks, deterministic=True
            )
        synchronize(device)
        policy_time_ms = (perf_counter() - policy_started) * 1000.0
        proposed_node = env._decode_target_node(int(tier), int(node))

        predictor_before_step = predictor_probe.snapshot()
        state_calls_before = len(state_records)
        mask_calls_before = len(mask_records)
        step_started = perf_counter()
        next_state, _, done, next_masks, info = env.step(tier, node)
        step_total_time_ms = (perf_counter() - step_started) * 1000.0
        predictor_after_step = predictor_probe.snapshot()
        if len(state_records) != state_calls_before + 1:
            raise RuntimeError("Expected exactly one state-generation call per step.")
        if len(mask_records) != mask_calls_before + 1:
            raise RuntimeError("Expected exactly one next-mask call per step.")
        next_state_record = state_records[-1]
        next_mask_record = mask_records[-1]
        step_predictor_delta = PredictorProbe.delta(
            predictor_before_step, predictor_after_step
        )
        transition_predictor_ms = max(
            0.0,
            float(step_predictor_delta["predictor_time_ms"])
            - float(next_state_record["predictor_time_ms"])
            - float(next_mask_record["predictor_time_ms"]),
        )
        transition_time_ms = max(
            0.0,
            step_total_time_ms
            - float(next_state_record["wall_time_ms"])
            - float(next_mask_record["wall_time_ms"]),
        )

        predictor_state_ms = float(current_state_record["predictor_time_ms"])
        predictor_mask_ms = float(current_mask_record["predictor_time_ms"])
        predictor_inference_ms = predictor_state_ms + predictor_mask_ms
        mask_generation_ms = float(
            current_mask_record.get(
                "mask_generation_time_ms", current_mask_record["wall_time_ms"]
            )
        )
        state_generation_ms = float(current_state_record["wall_time_ms"])
        online_decision_ms = (
            state_generation_ms
            + float(current_mask_record["wall_time_ms"])
            + state_tensor_time_ms
            + policy_time_ms
        )
        current_rss = process_rss_bytes()
        rss_peak = max(rss_peak, current_rss)
        if device.type == "cuda":
            cuda_allocated_sample_mb = torch.cuda.memory_allocated(device) / (
                1024.0**2
            )
            cuda_reserved_sample_mb = torch.cuda.memory_reserved(device) / (
                1024.0**2
            )
        else:
            cuda_allocated_sample_mb = float("nan")
            cuda_reserved_sample_mb = float("nan")
        decision_metrics = {
            "predictor_inference_time_ms": predictor_inference_ms,
            "predictor_state_time_ms": predictor_state_ms,
            "predictor_mask_time_ms": predictor_mask_ms,
            "predictor_transition_time_ms": transition_predictor_ms,
            "milp_solve_time_ms": float(
                current_mask_record.get("milp_solve_time_ms", float("nan"))
            ),
            "milp_pruner_time_ms": float(
                current_mask_record.get("milp_pruner_time_ms", float("nan"))
            ),
            "mask_generation_time_ms": mask_generation_ms,
            "state_generation_time_ms": state_generation_ms,
            "state_tensor_time_ms": state_tensor_time_ms,
            "policy_inference_time_ms": policy_time_ms,
            "online_decision_time_ms": online_decision_ms,
            "environment_transition_time_ms": transition_time_ms,
            "environment_step_total_time_ms": step_total_time_ms,
            "process_rss_sample_mb": current_rss / (1024.0**2),
            "cuda_allocated_sample_mb": cuda_allocated_sample_mb,
            "cuda_reserved_sample_mb": cuda_reserved_sample_mb,
        }

        decisions += 1
        timing_included = decisions > args.warmup_decisions
        if timing_included:
            timing_decisions += 1
            for metric, value in decision_metrics.items():
                values[metric].append(float(value))
                scenario_values[metric].append(float(value))

        executed_specs = env.node_specs[str(info.get("executed_node", proposed_node))]
        decision_row: Dict[str, object] = {
            "scenario": scenario,
            "edge_gpu_capacity_factor": edge_gpu_factor,
            "evaluation_seed": evaluation_seed,
            "episode": episode + 1,
            "decision": decisions,
            "timing_included": timing_included,
            "task_id": task.unique_id,
            "workflow_id": task.workflow_id,
            "privacy_level": int(task.privacy_level),
            "proposed_node": proposed_node,
            "executed_node": info.get("executed_node", proposed_node),
            "executed_tier": executed_specs["type"],
            "executed_node_gpu_capacity": float(executed_specs["gpu_mem"]),
            "executed_node_gpu_request_fraction": (
                float(task.req_gpu_mem) / max(float(executed_specs["gpu_mem"]), 1e-12)
            ),
            **binding,
            **decision_metrics,
            "predictor_calls_state": int(
                current_state_record["predictor_calls"]
            ),
            "predictor_calls_mask": int(current_mask_record["predictor_calls"]),
            "predictor_cache_hits_state": int(
                current_state_record["predictor_cache_hits"]
            ),
            "predictor_cache_hits_mask": int(
                current_mask_record["predictor_cache_hits"]
            ),
        }
        if decision_writer is not None:
            decision_writer.writerow(decision_row)
        if (
            bool(binding["gpu_constraint_binding"])
            and len(binding_events) < args.max_binding_events
        ):
            binding_events.append(decision_row)

        final_info = dict(info)
        state, masks = next_state, next_masks
        if args.max_decisions is not None and decisions >= args.max_decisions:
            break

    truncated = not done
    scenario_counts["episodes"] += 1
    scenario_counts["decisions"] += decisions
    scenario_counts["timing_decisions"] += timing_decisions
    scenario_counts["gpu_task_decisions"] += gpu_task_decisions
    scenario_counts["gpu_binding_decisions"] += gpu_binding_decisions
    scenario_counts["gpu_only_blocked_actions"] += gpu_only_blocked_actions
    scenario_counts["truncated_episodes"] += int(truncated)

    if device.type == "cuda":
        synchronize(device)
        cuda_peak_allocated = int(torch.cuda.max_memory_allocated(device))
        cuda_peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        cuda_peak_allocated = 0
        cuda_peak_reserved = 0

    row: Dict[str, object] = {
        "scenario": scenario,
        "edge_gpu_capacity_factor": edge_gpu_factor,
        "evaluation_seed": evaluation_seed,
        "checkpoint_seed": int(config.seed),
        "episode": episode + 1,
        "decisions": decisions,
        "timing_decisions": timing_decisions,
        "warmup_decisions": args.warmup_decisions,
        "truncated": truncated,
        "tasks": len(env.task_map),
        "workflows": len(env.workflow_tasks),
        "gpu_task_decisions": gpu_task_decisions,
        "gpu_binding_decisions": gpu_binding_decisions,
        "gpu_binding_decision_rate": gpu_binding_decisions / max(decisions, 1),
        "gpu_only_blocked_actions": gpu_only_blocked_actions,
        "dynamic_gpu_busy_block_events": 0,
        "process_rss_baseline_mb": rss_baseline / (1024.0**2),
        "process_rss_sampled_peak_mb": rss_peak / (1024.0**2),
        "process_rss_sampled_peak_delta_mb": (rss_peak - rss_baseline)
        / (1024.0**2),
        "cuda_measured": device.type == "cuda",
        "cuda_baseline_allocated_mb": cuda_baseline_allocated / (1024.0**2),
        "cuda_baseline_reserved_mb": cuda_baseline_reserved / (1024.0**2),
        "cuda_peak_allocated_mb": cuda_peak_allocated / (1024.0**2),
        "cuda_peak_reserved_mb": cuda_peak_reserved / (1024.0**2),
        "avg_wf_time": float(final_info.get("avg_wf_time", 0.0)),
        "ddl_rate": float(final_info.get("ddl_rate", 0.0)),
        **capacity_audit,
    }
    add_metric_stats(row, values)
    return runtime, row


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.slack_model = args.slack_model.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Frozen Full checkpoint not found: {args.checkpoint}")
    if not args.slack_model.exists():
        raise FileNotFoundError(f"Slack predictor not found: {args.slack_model}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    checkpoint_payload = safe_torch_load(args.checkpoint, torch.device("cpu"))
    args.eval_episodes = max(args.nominal_episodes, args.stress_episodes)
    args.max_steps = args.max_decisions
    config = checkpoint_config(checkpoint_payload, args)
    if config.variant != "full":
        raise ValueError(f"Expected Full H-MARL checkpoint, got {config.variant!r}.")
    switches = config.switches
    if not (
        switches.hierarchical
        and switches.slack_predictor
        and switches.milp_pruning
        and switches.action_mask
    ):
        raise ValueError("Checkpoint is not the complete Full H-MARL method.")
    runtime_payload = checkpoint_payload.get("runtime")
    if not isinstance(runtime_payload, Mapping):
        raise ValueError("Checkpoint has no policy runtime payload.")
    stored_policies = runtime_payload.get("policies")
    if not isinstance(stored_policies, Mapping) or not stored_policies:
        raise ValueError("Checkpoint runtime has no actor policy states.")
    policy_state: MutableMapping[str, object] = dict(stored_policies)
    del checkpoint_payload, runtime_payload, stored_policies
    gc.collect()

    decision_handle = None
    decision_writer = None
    decision_fields = [
        "scenario",
        "edge_gpu_capacity_factor",
        "evaluation_seed",
        "episode",
        "decision",
        "timing_included",
        "task_id",
        "workflow_id",
        "privacy_level",
        "proposed_node",
        "executed_node",
        "executed_tier",
        "executed_node_gpu_capacity",
        "executed_node_gpu_request_fraction",
        "task_gpu_request",
        "task_uses_gpu",
        "non_gpu_candidate_count",
        "gpu_feasible_candidate_count",
        "gpu_only_blocked_count",
        "gpu_constraint_binding",
        "gpu_only_blocked_pi",
        "gpu_only_blocked_edge",
        "gpu_only_blocked_cloud",
        "gpu_only_blocked_nodes",
        *TIMING_METRICS,
        "predictor_calls_state",
        "predictor_calls_mask",
        "predictor_cache_hits_state",
        "predictor_cache_hits_mask",
    ]
    if args.write_decision_csv:
        decision_handle = (
            args.output_dir / "runtime_gpu_decisions.csv"
        ).open("w", encoding="utf-8-sig", newline="")
        decision_writer = csv.DictWriter(decision_handle, fieldnames=decision_fields)
        decision_writer.writeheader()

    episode_rows: List[Dict[str, object]] = []
    binding_events: List[Dict[str, object]] = []
    scenario_values: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    scenario_counts: Dict[str, Dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    runtime: PolicyRuntime | None = None

    run_specs = [
        ("nominal", 1.0, seed, args.nominal_episodes)
        for seed in args.evaluation_seeds
    ]
    run_specs.append(
        (
            f"edge_gpu_capacity_{args.edge_gpu_stress_factor:g}x",
            args.edge_gpu_stress_factor,
            args.stress_seed,
            args.stress_episodes,
        )
    )

    try:
        for scenario, factor, seed, episodes in run_specs:
            for episode in range(episodes):
                print(
                    f"Profiling {scenario}: eval_seed={seed}, "
                    f"episode={episode + 1}/{episodes}"
                )
                runtime, episode_row = run_profiled_episode(
                    args=args,
                    config=config,
                    policy_state=policy_state,
                    runtime=runtime,
                    device=device,
                    evaluation_seed=int(seed),
                    episode=episode,
                    scenario=scenario,
                    edge_gpu_factor=float(factor),
                    scenario_values=scenario_values[scenario],
                    scenario_counts=scenario_counts[scenario],
                    binding_events=binding_events,
                    decision_writer=decision_writer,
                )
                episode_rows.append(episode_row)
    finally:
        if decision_handle is not None:
            decision_handle.close()

    scenario_rows: List[Dict[str, object]] = []
    for scenario, factor, _, _ in run_specs:
        if any(row["scenario"] == scenario for row in scenario_rows):
            continue
        counts = scenario_counts[scenario]
        decisions = int(counts["decisions"])
        selected_episodes = [
            row for row in episode_rows if row["scenario"] == scenario
        ]
        row: Dict[str, object] = {
            "scenario": scenario,
            "edge_gpu_capacity_factor": factor,
            "episodes": int(counts["episodes"]),
            "decisions": decisions,
            "timing_decisions": int(counts["timing_decisions"]),
            "truncated_episodes": int(counts["truncated_episodes"]),
            "gpu_task_decisions": int(counts["gpu_task_decisions"]),
            "gpu_task_decision_rate": counts["gpu_task_decisions"]
            / max(decisions, 1),
            "gpu_binding_decisions": int(counts["gpu_binding_decisions"]),
            "gpu_binding_decision_rate": counts["gpu_binding_decisions"]
            / max(decisions, 1),
            "gpu_only_blocked_actions": int(counts["gpu_only_blocked_actions"]),
            "dynamic_gpu_busy_block_events": 0,
            "process_rss_sampled_peak_mb": max(
                float(item["process_rss_sampled_peak_mb"])
                for item in selected_episodes
            ),
            "cuda_peak_allocated_mb": max(
                float(item["cuda_peak_allocated_mb"]) for item in selected_episodes
            ),
            "cuda_peak_reserved_mb": max(
                float(item["cuda_peak_reserved_mb"]) for item in selected_episodes
            ),
        }
        add_metric_stats(row, scenario_values[scenario])
        scenario_rows.append(row)

    nominal = next(row for row in scenario_rows if row["scenario"] == "nominal")
    stress = next(row for row in scenario_rows if row["scenario"] != "nominal")
    if int(nominal["gpu_binding_decisions"]) > 0:
        gpu_verdict = "gpu_memory_binds_in_nominal_traces"
    elif int(stress["gpu_binding_decisions"]) > 0:
        gpu_verdict = "nominal_never_binds_but_edge_gpu_stress_does"
    else:
        gpu_verdict = "gpu_memory_never_binds_even_in_edge_gpu_stress"

    expected_episodes = (
        len(args.evaluation_seeds) * args.nominal_episodes
        + args.stress_episodes
    )
    canonical_protocol = (
        args.checkpoint == DEFAULT_CHECKPOINT.expanduser().resolve()
        and args.slack_model == DEFAULT_PREDICTOR.expanduser().resolve()
        and args.evaluation_seeds == [42, 52, 62]
        and args.nominal_episodes == 1
        and args.stress_seed == 42
        and args.stress_episodes == 1
        and args.edge_gpu_stress_factor == 0.10
        and int(config.seed) == 42
        and str(config.implementation_version) == "e6-deadline-v2"
        and int(config.num_users) == 100
        and tuple(config.test_templates) == (4,)
    )
    complete = (
        args.max_decisions is None
        and len(episode_rows) == expected_episodes
        and all(not bool(row["truncated"]) for row in episode_rows)
        and all(int(row["timing_decisions"]) > 0 for row in episode_rows)
        and all(
            int(row[f"{metric}_count"]) > 0
            for row in scenario_rows
            for metric in (
                "predictor_inference_time_ms",
                "milp_solve_time_ms",
                "process_rss_sample_mb",
            )
        )
        and (
            device.type != "cuda"
            or all(
                int(row["cuda_allocated_sample_mb_count"]) > 0
                for row in scenario_rows
            )
        )
        and int(stress["gpu_binding_decisions"]) > 0
        and canonical_protocol
    )

    result_output_dir = args.output_dir
    if args.require_complete and not complete:
        result_output_dir = args.output_dir.with_name(
            f"{args.output_dir.name}_incomplete"
        )
        result_output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(result_output_dir / "runtime_gpu_episode_summary.csv", episode_rows)
    write_csv(result_output_dir / "runtime_gpu_scenario_summary.csv", scenario_rows)
    write_csv(result_output_dir / "runtime_gpu_binding_events.csv", binding_events)
    metadata = {
        "reviewer_issue": 10,
        "requested_output_directory": str(args.output_dir),
        "published_output_directory": str(result_output_dir),
        "protocol": (
            "deterministic frozen Full H-MARL evaluation; no training or "
            "parameter updates"
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_seed": int(config.seed),
        "evaluation_seeds": args.evaluation_seeds,
        "held_out_templates": list(config.test_templates),
        "nominal_episodes_per_seed": args.nominal_episodes,
        "stress_seed": args.stress_seed,
        "stress_episodes": args.stress_episodes,
        "edge_gpu_stress_factor": args.edge_gpu_stress_factor,
        "warmup_decisions": args.warmup_decisions,
        "device": str(device),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "memory_definition": (
            "decision-boundary sampled process RSS plus actor-only PyTorch CUDA "
            "allocated/reserved peaks when CUDA is used for policy inference; "
            "checkpoint training state and critic weights are not retained on CUDA"
        ),
        "gpu_constraint_definition": (
            "a decision is marked when GPU memory removes at least one node that "
            "privacy, CPU, and RAM otherwise permit; this does not imply every "
            "action is infeasible or that the final MILP choice changes"
        ),
        "dynamic_gpu_contention_scope": (
            "not modeled: the current serial single-server environment queues "
            "tasks and restores full resource capacity at each candidate start"
        ),
        "gpu_verdict": gpu_verdict,
        "complete": complete,
        "canonical_protocol": canonical_protocol,
        "expected_episodes": expected_episodes,
        "episode_summary": episode_rows,
        "scenario_summary": scenario_rows,
        "binding_event_samples": binding_events,
        "argv": sys.argv,
    }
    (result_output_dir / "runtime_gpu_summary.json").write_text(
        json.dumps(json_safe(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nRuntime/GPU diagnostics completed.")
    print(f"Output directory: {result_output_dir}")
    print(json.dumps(json_safe(scenario_rows), ensure_ascii=False, indent=2))
    print(f"GPU verdict: {gpu_verdict}")
    if args.require_complete and not complete:
        raise RuntimeError(
            "Run completed but did not satisfy the locked four-episode protocol; "
            f"inspect {result_output_dir / 'runtime_gpu_summary.json'}."
        )


if __name__ == "__main__":
    main()
