import os
import subprocess
import sys
from datetime import datetime


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_LOGIC_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
METHODS_ROOT = os.path.join(TRAINING_LOGIC_ROOT, "03_methods")
RESULTS_ROOT = os.path.join(TRAINING_LOGIC_ROOT, "06_results_archive")
RUNNER_LOG_DIR = os.path.join(RESULTS_ROOT, "experiment_runners", "E2_load_robustness")

# Edit these parameters directly before running this file.
RUN_CONFIG = {
    "method": "hyperjet",
    "num_users": 150,
    "seed": 42,
    "run_tag_suffix": None,
}

METHOD_SCRIPTS = {
    "custom_hmarl": os.path.join(METHODS_ROOT, "custom_hmarl", "train_hierarchical.py"),
    "hyperjet": os.path.join(METHODS_ROOT, "baseline_hyperjet", "train_hyperjet.py"),
    "maa2c": os.path.join(METHODS_ROOT, "baseline_maa2c", "train_maa2c.py"),
    "maddpg": os.path.join(METHODS_ROOT, "baseline_maddpg", "train_maddpg.py"),
    "mappo": os.path.join(METHODS_ROOT, "baseline_mappo", "train_mappo.py"),
}


def get_run_config():
    method = str(RUN_CONFIG["method"]).strip()
    num_users = int(RUN_CONFIG["num_users"])
    seed = int(RUN_CONFIG.get("seed", 42))
    run_tag_suffix = RUN_CONFIG.get("run_tag_suffix")
    if run_tag_suffix is not None:
        run_tag_suffix = str(run_tag_suffix).strip() or None

    if method not in METHOD_SCRIPTS:
        raise ValueError(
            f"Unsupported method: {method}. "
            f"Supported methods: {', '.join(sorted(METHOD_SCRIPTS.keys()))}"
        )
    if num_users <= 0:
        raise ValueError("num_users must be a positive integer.")

    return {
        "method": method,
        "num_users": num_users,
        "seed": seed,
        "run_tag_suffix": run_tag_suffix,
    }


def ensure_dirs():
    os.makedirs(RUNNER_LOG_DIR, exist_ok=True)


def build_env(num_users, seed, run_tag_suffix):
    env = os.environ.copy()
    env["TRAIN_NUM_USERS"] = str(num_users)
    env["TRAIN_SEED"] = str(seed)
    env["TRAIN_RUN_TAG_SUFFIX"] = run_tag_suffix or f"_e2_{num_users}"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_method(method_name, script_path, num_users, seed, run_tag_suffix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner_log_path = os.path.join(
        RUNNER_LOG_DIR,
        f"{method_name}_{num_users}users_seed{seed}_{timestamp}.log",
    )
    env = build_env(num_users, seed, run_tag_suffix)

    print(f"[START] {method_name} @ {num_users} users", flush=True)
    print(f"[SEED] {seed}", flush=True)
    print(f"[SCRIPT] {script_path}", flush=True)
    print(f"[RUN TAG SUFFIX] {env['TRAIN_RUN_TAG_SUFFIX']}", flush=True)
    print(f"[PYTHON] {sys.executable}", flush=True)
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
    config = get_run_config()

    script_path = METHOD_SCRIPTS[config["method"]]
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Training script not found: {script_path}")

    print("E2 single-run launcher is starting.", flush=True)
    print(f"Training logic root: {TRAINING_LOGIC_ROOT}", flush=True)
    print(f"Results root: {RESULTS_ROOT}", flush=True)
    print(f"Run config: {config}", flush=True)

    run_method(
        method_name=config["method"],
        script_path=script_path,
        num_users=config["num_users"],
        seed=config["seed"],
        run_tag_suffix=config["run_tag_suffix"],
    )

    print("Single E2 run finished successfully.", flush=True)


if __name__ == "__main__":
    main()
