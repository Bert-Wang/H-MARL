import csv
import json
import os
import subprocess
import sys
from datetime import datetime


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_LOGIC_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
METHODS_ROOT = os.path.join(TRAINING_LOGIC_ROOT, "03_methods")
RESULTS_ROOT = os.path.join(TRAINING_LOGIC_ROOT, "06_results_archive")
RUNNER_LOG_DIR = os.path.join(RESULTS_ROOT, "experiment_runners", "E1_main_comparison_100_users")
SUMMARY_ROOT = os.path.join(RESULTS_ROOT, "summaries")
MODEL_ROOT = os.path.join(RESULTS_ROOT, "models")
LOG_ROOT = os.path.join(RESULTS_ROOT, "logs")
FIGURE_ROOT = os.path.join(RESULTS_ROOT, "figures", "training_curves")
EXPECTED_EPOCHS = 1000

METHOD_SCRIPTS = [
    {
        "method": "custom_hmarl",
        "script": os.path.join(METHODS_ROOT, "custom_hmarl", "train_hierarchical.py"),
        "run_tag": "hierarchical",
        "figures": [
            "reward_total_{run_tag}.png",
            "reward_components_{run_tag}.png",
            "metrics_plot_{run_tag}.png",
        ],
    },
    {
        "method": "hyperjet",
        "script": os.path.join(METHODS_ROOT, "baseline_hyperjet", "train_hyperjet.py"),
        "run_tag": "hyperjet",
        "figures": [
            "training_curve_{run_tag}.png",
        ],
    },
    {
        "method": "maa2c",
        "script": os.path.join(METHODS_ROOT, "baseline_maa2c", "train_maa2c.py"),
        "run_tag": "maa2c",
        "figures": [
            "reward_total_{run_tag}.png",
            "reward_components_{run_tag}.png",
            "metrics_plot_{run_tag}.png",
        ],
    },
    {
        "method": "maddpg",
        "script": os.path.join(METHODS_ROOT, "baseline_maddpg", "train_maddpg.py"),
        "run_tag": "maddpg",
        "figures": [
            "training_curves_{run_tag}.png",
        ],
    },
    {
        "method": "mappo",
        "script": os.path.join(METHODS_ROOT, "baseline_mappo", "train_mappo.py"),
        "run_tag": "mappo",
        "figures": [
            "reward_total_{run_tag}.png",
            "reward_components_{run_tag}.png",
            "metrics_plot_{run_tag}.png",
        ],
    },
]


def ensure_dirs():
    os.makedirs(RUNNER_LOG_DIR, exist_ok=True)


def is_nonempty_file(file_path):
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0


def read_summary_json(summary_json_path):
    with open(summary_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def inspect_epoch_metrics(epoch_metrics_path):
    with open(epoch_metrics_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return False, "epoch_metrics CSV 没有训练记录"

    try:
        last_epoch = int(rows[-1]["epoch"])
    except (KeyError, TypeError, ValueError):
        return False, "epoch_metrics CSV 的 epoch 列无效"

    if len(rows) != EXPECTED_EPOCHS:
        return False, f"epoch_metrics CSV 只有 {len(rows)} 行训练记录，预期 {EXPECTED_EPOCHS} 行"

    if last_epoch != EXPECTED_EPOCHS:
        return False, f"epoch_metrics CSV 最后一轮是 {last_epoch}，预期 {EXPECTED_EPOCHS}"

    return True, ""


def build_expected_artifacts(method_config):
    method_name = method_config["method"]
    run_tag = method_config["run_tag"]
    figure_dir = os.path.join(FIGURE_ROOT, method_name)

    return {
        "log_file": os.path.join(LOG_ROOT, method_name, f"train_{run_tag}.log"),
        "checkpoint_file": os.path.join(MODEL_ROOT, method_name, f"checkpoint_{run_tag}.pth"),
        "best_model_file": os.path.join(MODEL_ROOT, method_name, f"best_model_{run_tag}.pth"),
        "epoch_metrics_file": os.path.join(SUMMARY_ROOT, method_name, f"epoch_metrics_{run_tag}.csv"),
        "summary_json": os.path.join(SUMMARY_ROOT, method_name, f"run_summary_{run_tag}.json"),
        "figure_files": [
            os.path.join(figure_dir, figure_name.format(run_tag=run_tag))
            for figure_name in method_config["figures"]
        ],
    }


def inspect_completed_run(method_config):
    method_name = method_config["method"]
    run_tag = method_config["run_tag"]
    expected_artifacts = build_expected_artifacts(method_config)
    issues = []

    required_files = [
        ("log_file", expected_artifacts["log_file"]),
        ("checkpoint_file", expected_artifacts["checkpoint_file"]),
        ("best_model_file", expected_artifacts["best_model_file"]),
        ("epoch_metrics_file", expected_artifacts["epoch_metrics_file"]),
        ("summary_json", expected_artifacts["summary_json"]),
    ]

    for label, file_path in required_files:
        if not is_nonempty_file(file_path):
            issues.append(f"{label} 缺失或为空: {file_path}")

    for figure_path in expected_artifacts["figure_files"]:
        if not is_nonempty_file(figure_path):
            issues.append(f"figure_file 缺失或为空: {figure_path}")

    if is_nonempty_file(expected_artifacts["epoch_metrics_file"]):
        metrics_ok, metrics_message = inspect_epoch_metrics(expected_artifacts["epoch_metrics_file"])
        if not metrics_ok:
            issues.append(metrics_message)

    summary_data = None
    if is_nonempty_file(expected_artifacts["summary_json"]):
        try:
            summary_data = read_summary_json(expected_artifacts["summary_json"])
        except (json.JSONDecodeError, OSError) as exc:
            issues.append(f"summary_json 无法解析: {exc}")

    if summary_data is not None:
        if summary_data.get("method") != method_name:
            issues.append(f"summary_json 的 method={summary_data.get('method')}，预期 {method_name}")
        if summary_data.get("run_tag") != run_tag:
            issues.append(f"summary_json 的 run_tag={summary_data.get('run_tag')}，预期 {run_tag}")

        training = summary_data.get("training") or {}
        if training.get("epochs") != EXPECTED_EPOCHS:
            issues.append(f"summary_json 的 training.epochs={training.get('epochs')}，预期 {EXPECTED_EPOCHS}")

        artifacts = summary_data.get("artifacts") or {}
        for artifact_key in ("log_file", "checkpoint_file", "best_model_file", "epoch_metrics_file"):
            actual_path = artifacts.get(artifact_key)
            expected_path = expected_artifacts[artifact_key]
            if os.path.normcase(os.path.normpath(str(actual_path or ""))) != os.path.normcase(os.path.normpath(expected_path)):
                issues.append(f"summary_json 的 {artifact_key} 与当前预期路径不一致")

        actual_figure_files = artifacts.get("figure_files") or []
        normalized_actual_figures = {
            os.path.normcase(os.path.normpath(str(path))) for path in actual_figure_files
        }
        normalized_expected_figures = {
            os.path.normcase(os.path.normpath(path)) for path in expected_artifacts["figure_files"]
        }
        if normalized_actual_figures != normalized_expected_figures:
            issues.append("summary_json 的 figure_files 与当前预期不一致")

    return len(issues) == 0, issues


def run_method(method_config):
    method_name = method_config["method"]
    script_path = method_config["script"]
    is_complete, issues = inspect_completed_run(method_config)

    if is_complete:
        print(f"[SKIP] {method_name}: 已检测到完整结果，跳过训练。")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner_log_path = os.path.join(RUNNER_LOG_DIR, f"{method_name}_{timestamp}.log")

    print(f"[INCOMPLETE] {method_name}: 检测到结果不完整，继续训练。")
    for issue in issues:
        print(f"  - {issue}")
    print(f"[START] {method_name}: {script_path}")
    print(f"[RUNNER LOG] {runner_log_path}")

    with open(runner_log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, script_path],
            cwd=TRAINING_LOGIC_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"{method_name} failed with exit code {return_code}. See {runner_log_path}")

    print(f"[DONE] {method_name}")


def main():
    ensure_dirs()
    print("E1 Main Comparison (100 users) is starting.")
    print(f"Training logic root: {TRAINING_LOGIC_ROOT}")
    print(f"Results root: {RESULTS_ROOT}")

    for item in METHOD_SCRIPTS:
        run_method(item)

    print("All E1 methods finished successfully.")
    print("Method-specific outputs have been written to training_logic/06_results_archive.")


if __name__ == "__main__":
    main()
