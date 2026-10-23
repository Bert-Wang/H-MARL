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
RUNNER_LOG_DIR = os.path.join(RESULTS_ROOT, "experiment_runners", "E2_load_robustness")
SUMMARY_ROOT = os.path.join(RESULTS_ROOT, "summaries")
MODEL_ROOT = os.path.join(RESULTS_ROOT, "models")
LOG_ROOT = os.path.join(RESULTS_ROOT, "logs")
FIGURE_ROOT = os.path.join(RESULTS_ROOT, "figures", "training_curves")

LOADS = [50, 100, 150]
DEFAULT_SEED = 42
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


def build_expected_artifacts(method_config, num_users):
    method_name = method_config["method"]
    run_tag = f"{method_config['run_tag']}_e2_{num_users}"
    figure_dir = os.path.join(FIGURE_ROOT, method_name)

    return {
        "run_tag": run_tag,
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


def inspect_completed_run(method_config, num_users):
    method_name = method_config["method"]
    expected_artifacts = build_expected_artifacts(method_config, num_users)
    run_tag = expected_artifacts["run_tag"]
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

        load = summary_data.get("load") or {}
        if load.get("num_users") != num_users:
            issues.append(f"summary_json 的 load.num_users={load.get('num_users')}，预期 {num_users}")

        training = summary_data.get("training") or {}
        if training.get("epochs") != EXPECTED_EPOCHS:
            issues.append(f"summary_json 的 training.epochs={training.get('epochs')}，预期 {EXPECTED_EPOCHS}")
        if training.get("seed") != DEFAULT_SEED:
            issues.append(f"summary_json 的 training.seed={training.get('seed')}，预期 {DEFAULT_SEED}")

    return len(issues) == 0, issues


def build_env(num_users):
    env = os.environ.copy()
    env["TRAIN_NUM_USERS"] = str(num_users)
    env["TRAIN_SEED"] = str(DEFAULT_SEED)
    env["TRAIN_RUN_TAG_SUFFIX"] = f"_e2_{num_users}"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_method(method_config, num_users):
    method_name = method_config["method"]
    script_path = method_config["script"]
    is_complete, issues = inspect_completed_run(method_config, num_users)

    if is_complete:
        print(f"[SKIP] {method_name} @ {num_users} users: 已检测到完整结果，跳过训练。", flush=True)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner_log_path = os.path.join(
        RUNNER_LOG_DIR,
        f"{method_name}_{num_users}users_{timestamp}.log",
    )
    env = build_env(num_users)

    print(f"[INCOMPLETE] {method_name} @ {num_users} users: 检测到结果不完整，继续训练。", flush=True)
    for issue in issues:
        print(f"  - {issue}", flush=True)
    print(f"[START] {method_name} @ {num_users} users", flush=True)
    print(f"[SCRIPT] {script_path}", flush=True)
    print(f"[RUN TAG SUFFIX] {env['TRAIN_RUN_TAG_SUFFIX']}", flush=True)
    print(f"[RUNNER LOG] {runner_log_path}", flush=True)

    with open(runner_log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, script_path],
            cwd=TRAINING_LOGIC_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )

        for line in iter(process.stdout.readline, ""):
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"{method_name} failed at {num_users} users with exit code {return_code}. "
            f"See {runner_log_path}"
        )

    print(f"[DONE] {method_name} @ {num_users} users", flush=True)


def main():
    ensure_dirs()
    print("E2 Load Robustness is starting.", flush=True)
    print(f"Training logic root: {TRAINING_LOGIC_ROOT}", flush=True)
    print(f"Results root: {RESULTS_ROOT}", flush=True)
    print(f"Loads: {LOADS}", flush=True)
    print(f"Seed: {DEFAULT_SEED}", flush=True)

    for num_users in LOADS:
        print(f"===== Load {num_users} users =====", flush=True)
        for item in METHOD_SCRIPTS:
            run_method(item, num_users)

    print("All E2 load-robustness runs finished successfully.", flush=True)
    print("Method-specific outputs have been written to training_logic/06_results_archive.", flush=True)


if __name__ == "__main__":
    main()
