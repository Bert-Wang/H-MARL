from __future__ import annotations

"""Train and evaluate the privacy-relaxed H-MARL ablation.

The existing ``w/o Mask`` variant is not a privacy ablation: execution still
enforces ``task.allowed_nodes``.  This script implements the reviewer-requested
comparison instead:

* Full H-MARL keeps the original per-task privacy admissible set.
* Privacy-relaxed H-MARL may schedule on the full node set, subject to the same
  resource constraints, MILP pruning, predictor, reward, DAG, and deadlines.
* Reference completions and deadlines are always computed with the original
  strict privacy sets, so the relaxed variant does not receive easier deadlines.

The script leaves the production environment and trainer untouched.  It
temporarily injects a subclass only while training the relaxed variant, then
performs a deterministic privacy audit for both variants.
"""

import argparse
import csv
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent


def locate_method_parent() -> Path:
    """Find the directory that directly contains ``paper_method_hmarl``.

    Linux deployments are often copied under an additional project directory,
    while Windows development uses the canonical ``training_logic/03_methods``
    layout.  Search both layouts and allow an explicit override.
    """

    candidates: List[Path] = []
    override = os.environ.get("HMARL_METHOD_PARENT")
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(SCRIPT_DIR.parents[1] / "03_methods")
    for ancestor in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        candidates.extend(
            (
                ancestor / "03_methods",
                ancestor / "training_logic" / "03_methods",
            )
        )

    checked: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        checked.append(resolved)
        package = resolved / "paper_method_hmarl"
        if (package / "config.py").is_file() and (
            package / "environment.py"
        ).is_file():
            return resolved

    checked_text = "\n  - ".join(str(path) for path in checked)
    raise ModuleNotFoundError(
        "Cannot locate the paper_method_hmarl source package.\n"
        "Checked:\n  - "
        f"{checked_text}\n"
        "Copy training_logic/03_methods/paper_method_hmarl to the Linux "
        "project, or set HMARL_METHOD_PARENT to the directory that contains "
        "paper_method_hmarl."
    )


METHOD_PARENT = locate_method_parent()
TRAINING_LOGIC_ROOT = METHOD_PARENT.parent
if str(METHOD_PARENT) not in sys.path:
    sys.path.insert(0, str(METHOD_PARENT))

from paper_method_hmarl.config import ExperimentConfig, MethodSwitches  # noqa: E402
from paper_method_hmarl.environment import PaperMethodEnvironment, PaperTask  # noqa: E402
from paper_method_hmarl.milp_pruner import MilpPrunerConfig  # noqa: E402
from paper_method_hmarl import trainer as trainer_module  # noqa: E402
from paper_method_hmarl.trainer import (  # noqa: E402
    PolicyRuntime,
    load_predictor,
    resolve_device,
    run_episode,
    safe_torch_load,
    set_random_seed,
)


DEFAULT_PREDICTOR = (
    TRAINING_LOGIC_ROOT
    / "06_results_archive"
    / "E6_paper_method_ablation"
    / "shared"
    / "slack_histgbrt_remaining_v2.joblib"
)
DEFAULT_ARCHIVE_RUNS = (
    TRAINING_LOGIC_ROOT
    / "06_results_archive"
    / "E6_paper_method_ablation"
    / "runs"
)
DEFAULT_OUTPUT_ROOT = (
    TRAINING_LOGIC_ROOT
    / "06_results_archive"
    / "E6_paper_method_ablation"
    / "privacy_ablation"
)

FULL_SWITCHES = MethodSwitches(
    hierarchical=True,
    slack_predictor=True,
    milp_pruning=True,
    action_mask=True,
)

STANDARD_METRICS = (
    "avg_reward",
    "avg_wf_time",
    "ddl_rate",
    "mean_deadline",
    "mean_reference_completion",
    "avg_critical_slack",
    "avg_pruning_rate",
    "invalid_action_rate",
    "correction_rate",
)

COMPARISON_CONFIG_FIELDS = (
    "num_users",
    "seed",
    "epochs",
    "learning_rate",
    "gamma",
    "hidden_dim",
    "entropy_beta",
    "entropy_beta_min",
    "entropy_decay",
    "grad_clip",
    "lr_step_size",
    "lr_gamma",
    "time_interval",
    "workload_jitter",
    "cross_workflow_edge_rate",
    "deadline_factor",
    "train_templates",
    "test_templates",
    "reward_weight_latency",
    "reward_weight_ddl",
    "reward_weight_slack",
    "invalid_action_penalty",
    "milp_slack_threshold",
    "milp_time_limit",
    "milp_strict",
    "milp_relax_top_k",
)

SEED_SUMMARY_FIELDS = (
    "variant",
    "privacy_mode",
    "seed",
    "checkpoint",
    "comparison_fingerprint",
    "episodes",
    "decisions",
    "truncated_episode_count",
    "avg_reward",
    "avg_wf_time",
    "ddl_rate",
    "mean_deadline",
    "mean_reference_completion",
    "avg_critical_slack",
    "avg_pruning_rate",
    "invalid_action_rate",
    "correction_rate",
    "privacy_compliance_rate",
    "privacy_violation_rate",
    "privacy_violation_count",
    "proposed_privacy_compliance_rate",
    "empty_strict_admissible_set_count",
    "empty_strict_admissible_set_rate",
    "empty_strict_resource_feasible_set_count",
    "empty_strict_resource_feasible_set_rate",
    "privacy_level_0_compliance_rate",
    "privacy_level_1_compliance_rate",
    "privacy_level_2_compliance_rate",
    "placement_pi_rate",
    "placement_edge_rate",
    "placement_cloud_rate",
)

PAIR_FIELDS = (
    "seed",
    "full_checkpoint",
    "relaxed_checkpoint",
    "full_avg_wf_time",
    "relaxed_avg_wf_time",
    "avg_wf_time_delta",
    "avg_wf_time_change_pct",
    "full_ddl_rate",
    "relaxed_ddl_rate",
    "ddl_rate_delta",
    "full_avg_critical_slack",
    "relaxed_avg_critical_slack",
    "avg_critical_slack_delta",
    "full_privacy_compliance_rate",
    "relaxed_privacy_compliance_rate",
    "privacy_compliance_delta",
)


class PrivacyAuditEnvironment(PaperMethodEnvironment):
    """Paper environment with optional privacy relaxation and audit records."""

    def __init__(self, *args, relax_privacy: bool, **kwargs) -> None:
        self.relax_privacy = bool(relax_privacy)
        self.strict_allowed_nodes: Dict[str, tuple[str, ...]] = {}
        self.privacy_records: List[Dict[str, object]] = []
        super().__init__(*args, **kwargs)

    def reset(self):
        self.privacy_records = []
        return super().reset()

    def _load_tasks(self) -> None:
        super()._load_tasks()
        self.strict_allowed_nodes = {
            task_id: tuple(task.allowed_nodes)
            for task_id, task in self.task_map.items()
        }

    def _initialize_graph_quantities(self) -> None:
        # Super computes policy-independent reference completions and deadlines
        # while every task still has its original strict privacy set.
        super()._initialize_graph_quantities()
        if self.relax_privacy:
            full_node_set = list(self.all_nodes)
            for task in self.task_map.values():
                task.allowed_nodes = list(full_node_set)

    def _resource_feasible_count(
        self, task: PaperTask, candidate_nodes: Sequence[str]
    ) -> int:
        count = 0
        for node in candidate_nodes:
            specs = self.node_specs[node]
            feasible = (
                float(specs["cpu"]) + 1e-9 >= task.core_count
                and float(specs["mem"]) + 1e-9 >= task.req_mem
                and float(specs["gpu_mem"]) + 1e-9 >= task.req_gpu_mem
            )
            count += int(feasible)
        return count

    def step(self, tier_action: int, node_action: int):
        if self.current_task is None:
            return super().step(tier_action, node_action)

        task = self.current_task
        task_id = task.unique_id
        privacy_level = int(task.privacy_level)
        strict_nodes = self.strict_allowed_nodes.get(task_id, tuple())
        proposed_node = self._decode_target_node(int(tier_action), int(node_action))
        strict_resource_feasible_count = self._resource_feasible_count(
            task, strict_nodes
        )

        next_state, reward, done, next_mask, info = super().step(
            tier_action, node_action
        )
        executed_node = str(info.get("executed_node", ""))
        proposed_compliant = proposed_node in strict_nodes
        executed_compliant = executed_node in strict_nodes
        executed_tier = (
            str(self.node_specs[executed_node]["type"])
            if executed_node in self.node_specs
            else ""
        )
        record = {
            "task_id": task_id,
            "workflow_id": task.workflow_id,
            "privacy_level": privacy_level,
            "strict_admissible_count": len(strict_nodes),
            "strict_resource_feasible_count": strict_resource_feasible_count,
            "relaxed_resource_feasible_count": self._resource_feasible_count(
                task, self.all_nodes
            ),
            "proposed_node": proposed_node,
            "executed_node": executed_node,
            "executed_tier": executed_tier,
            "proposed_privacy_compliant": proposed_compliant,
            "privacy_compliant": executed_compliant,
            "privacy_violation": not executed_compliant,
            "privacy_relaxed": self.relax_privacy,
        }
        self.privacy_records.append(record)
        info.update(record)
        return next_state, reward, done, next_mask, info


def build_audit_environment(
    config: ExperimentConfig,
    predictor,
    *,
    relax_privacy: bool,
    seed_offset: int = 0,
    evaluation: bool = False,
) -> PrivacyAuditEnvironment:
    return PrivacyAuditEnvironment(
        num_users=config.num_users,
        time_interval=config.time_interval,
        use_slack_predictor=config.switches.slack_predictor,
        use_milp_pruning=config.switches.milp_pruning,
        use_action_mask=config.switches.action_mask,
        slack_predictor=predictor,
        milp_config=MilpPrunerConfig(
            slack_threshold=config.milp_slack_threshold,
            time_limit=config.milp_time_limit,
            strict=config.milp_strict,
            relax_top_k=config.milp_relax_top_k,
        ),
        reward_weight_latency=config.reward_weight_latency,
        reward_weight_ddl=config.reward_weight_ddl,
        reward_weight_slack=config.reward_weight_slack,
        invalid_action_penalty=config.invalid_action_penalty,
        base_seed=config.seed + seed_offset,
        workload_jitter=config.workload_jitter,
        cross_workflow_edge_rate=config.cross_workflow_edge_rate,
        deadline_factor=config.deadline_factor,
        template_ids=config.test_templates if evaluation else config.train_templates,
        relax_privacy=relax_privacy,
    )


@contextmanager
def relaxed_training_environment():
    """Patch only the trainer module's factory for the duration of training."""

    original = trainer_module.build_environment

    def factory(config, predictor, *, seed_offset=0, evaluation=False):
        return build_audit_environment(
            config,
            predictor,
            relax_privacy=True,
            seed_offset=seed_offset,
            evaluation=evaluation,
        )

    trainer_module.build_environment = factory
    try:
        yield
    finally:
        trainer_module.build_environment = original


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Train Full and privacy-relaxed H-MARL, then report performance "
            "cost and privacy compliance under the 100-source condition."
        )
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--num-users", type=int, default=100, choices=[50, 100, 150])
    parser.add_argument("--train-templates", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--test-templates", type=int, nargs="+", default=[4])
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--deadline-factor", type=float, default=1.10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--slack-model", type=Path, default=DEFAULT_PREDICTOR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / f"run_{timestamp}",
    )
    parser.add_argument(
        "--full-mode",
        choices=("reuse-or-train", "train", "skip"),
        default="reuse-or-train",
        help=(
            "reuse-or-train uses a compatible archived Full checkpoint when "
            "available and trains missing seeds; train retrains all Full seeds."
        ),
    )
    parser.add_argument(
        "--full-checkpoint-template",
        default=None,
        help="Optional path template containing {seed} for Full checkpoints.",
    )
    parser.add_argument(
        "--reuse-relaxed",
        action="store_true",
        help="Reuse privacy_relaxed_seed*/best_model.pt in --output-dir/runs.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--write-decision-csv",
        action="store_true",
        help="Also save every audited scheduling decision (large file).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.seeds:
        raise ValueError("At least one seed is required.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.eval_episodes <= 0:
        raise ValueError("--eval-episodes must be positive.")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive when provided.")
    if args.deadline_factor < 1.0:
        raise ValueError("--deadline-factor must be at least 1.0.")
    train = set(args.train_templates)
    test = set(args.test_templates)
    if not train or not test or train.intersection(test):
        raise ValueError("Train/test templates must be non-empty and disjoint.")
    invalid = sorted((train | test) - set(range(5)))
    if invalid:
        raise ValueError(f"Template IDs must be in [0, 4], got {invalid}")
    if args.full_checkpoint_template and "{seed}" not in args.full_checkpoint_template:
        raise ValueError("--full-checkpoint-template must contain {seed}.")


def make_training_config(
    args: argparse.Namespace, *, seed: int, variant: str
) -> ExperimentConfig:
    return ExperimentConfig(
        implementation_version=(
            "e6-privacy-relaxed-v1" if variant == "privacy_relaxed" else "e6-deadline-v2"
        ),
        variant=variant,
        switches=FULL_SWITCHES,
        num_users=args.num_users,
        seed=int(seed),
        epochs=args.epochs,
        checkpoint_interval=args.checkpoint_interval,
        deadline_factor=args.deadline_factor,
        train_templates=tuple(args.train_templates),
        test_templates=tuple(args.test_templates),
        max_steps_per_episode=args.max_steps,
        eval_episodes=args.eval_episodes,
        device=args.device,
        run_tag=f"{variant}_{args.num_users}sources_seed{seed}",
        resume=args.resume,
        slack_model_path=str(args.slack_model.resolve()),
        results_root=str((args.output_dir / "runs").resolve()),
    )


def checkpoint_config(
    payload: Mapping[str, object], args: argparse.Namespace
) -> ExperimentConfig:
    raw = payload.get("config")
    if not isinstance(raw, Mapping):
        raise ValueError("Checkpoint is missing its ExperimentConfig payload.")
    allowed = {item.name for item in fields(ExperimentConfig)}
    values = {key: value for key, value in raw.items() if key in allowed}
    switches = values.get("switches", {})
    if isinstance(switches, Mapping):
        values["switches"] = MethodSwitches(**switches)
    elif not isinstance(switches, MethodSwitches):
        raise TypeError(f"Unsupported switches payload: {type(switches)!r}")
    values["device"] = args.device
    values["eval_episodes"] = args.eval_episodes
    values["max_steps_per_episode"] = args.max_steps
    if values["switches"].slack_predictor:
        values["slack_model_path"] = str(args.slack_model.resolve())
    return ExperimentConfig(**values)


def comparison_fingerprint(config: ExperimentConfig) -> Dict[str, object]:
    raw = config.to_dict()
    fingerprint = {key: raw[key] for key in COMPARISON_CONFIG_FIELDS}
    fingerprint["switches"] = raw["switches"]
    predictor_path = Path(str(config.slack_model_path)).expanduser().resolve()
    fingerprint["slack_model"] = str(predictor_path)
    return fingerprint


def fingerprint_text(config: ExperimentConfig) -> str:
    return json.dumps(
        comparison_fingerprint(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def validate_checkpoint_compatibility(
    checkpoint_path: Path,
    intended: ExperimentConfig,
    *,
    expected_variant: str,
) -> ExperimentConfig:
    device = resolve_device(intended.device)
    payload = safe_torch_load(checkpoint_path, device)
    raw = payload.get("config")
    if not isinstance(raw, Mapping):
        raise ValueError(f"Checkpoint has no ExperimentConfig: {checkpoint_path}")
    allowed = {item.name for item in fields(ExperimentConfig)}
    values = {key: value for key, value in raw.items() if key in allowed}
    switches = values.get("switches", {})
    if isinstance(switches, Mapping):
        values["switches"] = MethodSwitches(**switches)
    elif not isinstance(switches, MethodSwitches):
        raise TypeError(f"Unsupported switches payload: {type(switches)!r}")
    loaded = ExperimentConfig(**values)
    if loaded.variant != expected_variant:
        raise ValueError(
            f"Expected variant={expected_variant!r}, got {loaded.variant!r}: "
            f"{checkpoint_path}"
        )
    expected_fp = comparison_fingerprint(intended)
    loaded_fp = comparison_fingerprint(loaded)
    mismatches = {
        key: {"checkpoint": loaded_fp.get(key), "requested": expected_fp.get(key)}
        for key in expected_fp
        if loaded_fp.get(key) != expected_fp.get(key)
    }
    if mismatches:
        raise ValueError(
            "Checkpoint is not compatible with the requested paired ablation: "
            f"{checkpoint_path}\n{json.dumps(mismatches, ensure_ascii=False, indent=2)}"
        )
    return loaded


def archived_full_checkpoint(
    seed: int, num_users: int, template: str | None
) -> Sequence[Path]:
    if template:
        candidate = Path(template.format(seed=seed)).expanduser().resolve()
        return (candidate,) if candidate.exists() else tuple()
    return (
        DEFAULT_ARCHIVE_RUNS
        / f"ablation_full_{num_users}users_seed{seed}_deadline_v2"
        / "best_model.pt",
        DEFAULT_ARCHIVE_RUNS
        / f"ablation_full_{num_users}users_seed{seed}"
        / "best_model.pt",
    )


def train_full_if_needed(args: argparse.Namespace, seed: int) -> Path | None:
    if args.full_mode == "skip":
        return None
    config = make_training_config(args, seed=seed, variant="full")
    if args.full_mode == "reuse-or-train":
        for archived in archived_full_checkpoint(
            seed, args.num_users, args.full_checkpoint_template
        ):
            if not archived.exists():
                continue
            try:
                validate_checkpoint_compatibility(
                    archived.resolve(), config, expected_variant="full"
                )
            except (TypeError, ValueError) as error:
                print(f"Skipping incompatible Full checkpoint: {error}")
                continue
            print(f"Reusing Full checkpoint for seed {seed}: {archived}")
            return archived.resolve()
    print(f"Training Full H-MARL for seed {seed}...")
    run_dir = trainer_module.train_experiment(config)
    return (run_dir / "best_model.pt").resolve()


def train_relaxed(args: argparse.Namespace, seed: int) -> Path:
    config = make_training_config(args, seed=seed, variant="privacy_relaxed")
    expected = Path(config.results_root) / config.resolved_run_tag() / "best_model.pt"
    manifest_path = expected.parent / "privacy_mode.json"
    if args.reuse_relaxed:
        if not expected.exists():
            raise FileNotFoundError(f"Relaxed checkpoint not found: {expected}")
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Privacy provenance manifest not found: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("privacy_mode") != "relaxed" or not manifest.get(
            "relax_privacy"
        ):
            raise ValueError(f"Invalid privacy provenance manifest: {manifest_path}")
        validate_checkpoint_compatibility(
            expected.resolve(), config, expected_variant="privacy_relaxed"
        )
        print(f"Reusing privacy-relaxed checkpoint for seed {seed}: {expected}")
        return expected.resolve()
    print(f"Training privacy-relaxed H-MARL for seed {seed}...")
    with relaxed_training_environment():
        run_dir = trainer_module.train_experiment(config)
    trained_checkpoint = (run_dir / "best_model.pt").resolve()
    manifest_path.write_text(
        json.dumps(
            {
                "privacy_mode": "relaxed",
                "relax_privacy": True,
                "checkpoint": str(trained_checkpoint),
                "comparison_fingerprint": comparison_fingerprint(config),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return trained_checkpoint


def safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_privacy(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    decisions = len(records)
    compliant = sum(bool(row["privacy_compliant"]) for row in records)
    proposed_compliant = sum(
        bool(row["proposed_privacy_compliant"]) for row in records
    )
    empty_admissible = sum(
        int(row["strict_admissible_count"]) == 0 for row in records
    )
    empty_resource = sum(
        int(row["strict_resource_feasible_count"]) == 0 for row in records
    )
    result: Dict[str, object] = {
        "decisions": decisions,
        "privacy_compliance_rate": safe_rate(compliant, decisions),
        "privacy_violation_rate": safe_rate(decisions - compliant, decisions),
        "privacy_violation_count": decisions - compliant,
        "proposed_privacy_compliance_rate": safe_rate(
            proposed_compliant, decisions
        ),
        "empty_strict_admissible_set_count": empty_admissible,
        "empty_strict_admissible_set_rate": safe_rate(empty_admissible, decisions),
        "empty_strict_resource_feasible_set_count": empty_resource,
        "empty_strict_resource_feasible_set_rate": safe_rate(
            empty_resource, decisions
        ),
    }
    for level in (0, 1, 2):
        level_rows = [
            row for row in records if int(row["privacy_level"]) == level
        ]
        level_compliant = sum(
            bool(row["privacy_compliant"]) for row in level_rows
        )
        result[f"privacy_level_{level}_compliance_rate"] = safe_rate(
            level_compliant, len(level_rows)
        )
    for tier in ("pi", "edge", "cloud"):
        tier_count = sum(str(row["executed_tier"]) == tier for row in records)
        result[f"placement_{tier}_rate"] = safe_rate(tier_count, decisions)
    return result


def audit_checkpoint(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    privacy_mode: str,
    decision_rows: List[Dict[str, object]] | None,
) -> Dict[str, object]:
    relax_privacy = privacy_mode == "relaxed"
    device = resolve_device(args.device)
    payload = safe_torch_load(checkpoint_path, device)
    config = checkpoint_config(payload, args)
    predictor = load_predictor(config)
    set_random_seed(config.seed + 10000)
    env = build_audit_environment(
        config,
        predictor,
        relax_privacy=relax_privacy,
        seed_offset=10000,
        evaluation=True,
    )
    initial_state = env.reset()
    runtime = PolicyRuntime(config, len(initial_state), env.tier_sizes, device)
    if "runtime" not in payload:
        raise ValueError(f"Checkpoint has no policy runtime: {checkpoint_path}")
    runtime.load_state_dict(payload["runtime"], load_optimizers=False)
    for module in runtime.policies.values():
        module.eval()
    runtime.critic.eval()

    episode_rows: List[Dict[str, object]] = []
    all_records: List[Dict[str, object]] = []
    for episode in range(config.eval_episodes):
        metrics = run_episode(env, runtime, config, training=False)
        privacy = summarize_privacy(env.privacy_records)
        episode_row = {
            "variant": config.variant,
            "privacy_mode": privacy_mode,
            "seed": config.seed,
            "episode": episode + 1,
            **metrics,
            **privacy,
        }
        episode_rows.append(episode_row)
        if decision_rows is not None:
            for row in env.privacy_records:
                decision_rows.append(
                    {
                        "variant": config.variant,
                        "privacy_mode": privacy_mode,
                        "seed": config.seed,
                        "episode": episode + 1,
                        **row,
                    }
                )
        all_records.extend(env.privacy_records)

    summary: Dict[str, object] = {
        "variant": config.variant,
        "privacy_mode": privacy_mode,
        "seed": config.seed,
        "checkpoint": str(checkpoint_path),
        "comparison_fingerprint": fingerprint_text(config),
        "episodes": config.eval_episodes,
        "truncated_episode_count": sum(
            bool(row["truncated"]) for row in episode_rows
        ),
    }
    for metric in STANDARD_METRICS:
        values = np.asarray(
            [float(row[metric]) for row in episode_rows], dtype=np.float64
        )
        summary[metric] = float(values.mean())
    summary.update(summarize_privacy(all_records))
    return summary


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def paired_rows(seed_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    by_seed: MutableMapping[int, Dict[str, Mapping[str, object]]] = {}
    for row in seed_rows:
        by_seed.setdefault(int(row["seed"]), {})[str(row["privacy_mode"])] = row
    output: List[Dict[str, object]] = []
    for seed, modes in sorted(by_seed.items()):
        if "strict" not in modes or "relaxed" not in modes:
            continue
        full = modes["strict"]
        relaxed = modes["relaxed"]
        if full["comparison_fingerprint"] != relaxed["comparison_fingerprint"]:
            raise ValueError(
                f"Seed {seed} has mismatched strict/relaxed configuration fingerprints."
            )
        for metric in ("mean_deadline", "mean_reference_completion"):
            if not np.isclose(
                float(full[metric]),
                float(relaxed[metric]),
                rtol=1e-10,
                atol=1e-10,
            ):
                raise ValueError(
                    f"Seed {seed} has mismatched {metric}: "
                    f"Full={full[metric]}, relaxed={relaxed[metric]}"
                )
        full_wf = float(full["avg_wf_time"])
        relaxed_wf = float(relaxed["avg_wf_time"])
        output.append(
            {
                "seed": seed,
                "full_checkpoint": full["checkpoint"],
                "relaxed_checkpoint": relaxed["checkpoint"],
                "full_avg_wf_time": full_wf,
                "relaxed_avg_wf_time": relaxed_wf,
                "avg_wf_time_delta": relaxed_wf - full_wf,
                "avg_wf_time_change_pct": (
                    100.0 * (relaxed_wf - full_wf) / full_wf if full_wf else 0.0
                ),
                "full_ddl_rate": float(full["ddl_rate"]),
                "relaxed_ddl_rate": float(relaxed["ddl_rate"]),
                "ddl_rate_delta": float(relaxed["ddl_rate"])
                - float(full["ddl_rate"]),
                "full_avg_critical_slack": float(full["avg_critical_slack"]),
                "relaxed_avg_critical_slack": float(
                    relaxed["avg_critical_slack"]
                ),
                "avg_critical_slack_delta": float(
                    relaxed["avg_critical_slack"]
                )
                - float(full["avg_critical_slack"]),
                "full_privacy_compliance_rate": float(
                    full["privacy_compliance_rate"]
                ),
                "relaxed_privacy_compliance_rate": float(
                    relaxed["privacy_compliance_rate"]
                ),
                "privacy_compliance_delta": float(
                    relaxed["privacy_compliance_rate"]
                )
                - float(full["privacy_compliance_rate"]),
            }
        )
    return output


def aggregate_rows(seed_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    numeric = (
        "avg_reward",
        "avg_wf_time",
        "ddl_rate",
        "avg_critical_slack",
        "privacy_compliance_rate",
        "privacy_violation_rate",
        "empty_strict_admissible_set_rate",
        "empty_strict_resource_feasible_set_rate",
        "placement_pi_rate",
        "placement_edge_rate",
        "placement_cloud_rate",
    )
    for mode in ("strict", "relaxed"):
        selected = [row for row in seed_rows if row["privacy_mode"] == mode]
        if not selected:
            continue
        row: Dict[str, object] = {
            "privacy_mode": mode,
            "seed_count": len(selected),
        }
        for metric in numeric:
            values = np.asarray(
                [float(item[metric]) for item in selected], dtype=np.float64
            )
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        output.append(row)
    return output


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir = args.output_dir.resolve()
    args.slack_model = args.slack_model.resolve()
    if not args.slack_model.exists():
        raise FileNotFoundError(f"Slack predictor not found: {args.slack_model}")
    if args.output_dir.exists() and not (args.resume or args.reuse_relaxed):
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}. "
            "Use a new --output-dir, or pass --resume/--reuse-relaxed."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runs").mkdir(parents=True, exist_ok=True)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "reviewer problem 5: privacy-relaxed admissible set",
        "strict_definition": "original privacy-level-specific allowed_nodes",
        "relaxed_definition": "all Pi, edge, and cloud nodes subject to resource/MILP constraints",
        "deadline_control": "strict privacy reference schedule for both variants",
        "seeds": args.seeds,
        "epochs": args.epochs,
        "num_users": args.num_users,
        "train_templates": args.train_templates,
        "test_templates": args.test_templates,
        "eval_episodes": args.eval_episodes,
        "deadline_factor": args.deadline_factor,
        "max_steps": args.max_steps,
        "device": args.device,
        "slack_model": str(args.slack_model),
        "full_mode": args.full_mode,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    seed_rows: List[Dict[str, object]] = []
    decision_rows: List[Dict[str, object]] | None = (
        [] if args.write_decision_csv else None
    )
    checkpoint_index: List[Dict[str, object]] = []
    for seed in args.seeds:
        full_checkpoint = train_full_if_needed(args, int(seed))
        relaxed_checkpoint = train_relaxed(args, int(seed))
        checkpoint_index.append(
            {
                "seed": int(seed),
                "full_checkpoint": str(full_checkpoint) if full_checkpoint else "",
                "relaxed_checkpoint": str(relaxed_checkpoint),
            }
        )

        if full_checkpoint is not None:
            print(f"Auditing strict Full H-MARL for seed {seed}...")
            seed_rows.append(
                audit_checkpoint(
                    args=args,
                    checkpoint_path=full_checkpoint,
                    privacy_mode="strict",
                    decision_rows=decision_rows,
                )
            )
        print(f"Auditing privacy-relaxed H-MARL for seed {seed}...")
        seed_rows.append(
            audit_checkpoint(
                args=args,
                checkpoint_path=relaxed_checkpoint,
                privacy_mode="relaxed",
                decision_rows=decision_rows,
            )
        )

    pairs = paired_rows(seed_rows)
    aggregate = aggregate_rows(seed_rows)
    write_csv(
        args.output_dir / "privacy_ablation_seed_summary.csv",
        seed_rows,
        SEED_SUMMARY_FIELDS,
    )
    write_csv(
        args.output_dir / "privacy_ablation_paired_comparison.csv",
        pairs,
        PAIR_FIELDS,
    )
    write_csv(args.output_dir / "privacy_ablation_summary.csv", aggregate)
    write_csv(args.output_dir / "privacy_ablation_checkpoints.csv", checkpoint_index)
    if decision_rows is not None:
        write_csv(args.output_dir / "privacy_ablation_decisions.csv", decision_rows)

    payload = {
        "metadata": metadata,
        "per_seed": seed_rows,
        "paired_comparison": pairs,
        "aggregate": aggregate,
        "metric_definitions": {
            "privacy_compliance_rate": "executed placements inside the original strict privacy admissible set / all decisions",
            "privacy_violation_rate": "executed placements outside the original strict privacy admissible set / all decisions",
            "proposed_privacy_compliance_rate": "policy-proposed placements inside the original strict set / all decisions",
            "empty_strict_admissible_set_rate": "audited decisions whose task has an empty original privacy allowed_nodes / all audited decisions",
            "empty_strict_resource_feasible_set_rate": "audited decisions whose task has no statically CPU/RAM/GPU-feasible node inside its original privacy set / all audited decisions",
            "avg_wf_time_delta": "privacy-relaxed minus Full; negative means faster",
            "ddl_rate_delta": "privacy-relaxed minus Full; negative means fewer deadline violations",
            "privacy_compliance_delta": "privacy-relaxed minus Full; negative quantifies lost privacy compliance",
        },
    }
    (args.output_dir / "privacy_ablation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nPrivacy ablation completed.")
    print(f"Output directory: {args.output_dir}")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
