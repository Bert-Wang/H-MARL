import os
import subprocess
import sys
from datetime import datetime


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_LOGIC_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
METHODS_ROOT = os.path.join(TRAINING_LOGIC_ROOT, "03_methods")
RESULTS_ROOT = os.path.join(TRAINING_LOGIC_ROOT, "06_results_archive")
RUNNER_LOG_DIR = os.path.join(RESULTS_ROOT, "experiment_runners", "E5_ablation")


ABLATION_SCRIPTS = [
    {
        "name": "A0_full_hmarl",
        "script": os.path.join(METHODS_ROOT, "custom_hmarl", "train_hierarchical.py"),
        "run_tag_suffix": "_e5_full",
    },
    {
        "name": "A2_wo_slack",
        "script": os.path.join(METHODS_ROOT, "custom_hmarl", "train_hierarchical_abl_wo_slack.py"),
        "run_tag_suffix": "",
    },
    {
        "name": "A4_wo_ddl_reward",
        "script": os.path.join(METHODS_ROOT, "custom_hmarl", "train_hierarchical_abl_wo_ddl_reward.py"),
        "run_tag_suffix": "",
    },
    {
        "name": "A3_no_mask",
        "script": os.path.join(METHODS_ROOT, "custom_hmarl", "train_hierarchical_abl_no_mask.py"),
        "run_tag_suffix": "",
    },
    {
        "name": "A1_wo_hierarchy",
        "script": os.path.join(METHODS_ROOT, "custom_hmarl", "train_hierarchical_abl_wo_hierarchy.py"),
        "run_tag_suffix": "",
    },
]


def ensure_dirs():
    os.makedirs(RUNNER_LOG_DIR, exist_ok=True)


def build_env(run_tag_suffix):
    env = os.environ.copy()
    if run_tag_suffix:
        env["TRAIN_RUN_TAG_SUFFIX"] = run_tag_suffix
    else:
        env.pop("TRAIN_RUN_TAG_SUFFIX", None)
    return env


def run_script(item):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner_log_path = os.path.join(RUNNER_LOG_DIR, f"{item['name']}_{timestamp}.log")
    env = build_env(item["run_tag_suffix"])

    print(f"[START] {item['name']}")
    print(f"[SCRIPT] {item['script']}")
    if item["run_tag_suffix"]:
        print(f"[RUN TAG SUFFIX] {item['run_tag_suffix']}")
    print(f"[RUNNER LOG] {runner_log_path}")

    with open(runner_log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, item["script"]],
            cwd=TRAINING_LOGIC_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"{item['name']} failed with exit code {return_code}. See {runner_log_path}"
        )

    print(f"[DONE] {item['name']}")


def main():
    ensure_dirs()
    print("E5 Ablation main run is starting.")
    print(f"Training logic root: {TRAINING_LOGIC_ROOT}")
    print(f"Results root: {RESULTS_ROOT}")
    print("Execution order: A0 -> A2 -> A4 -> A3 -> A1")

    for item in ABLATION_SCRIPTS:
        run_script(item)

    print("All E5 ablation runs finished successfully.")
    print("Method-specific outputs have been written to training_logic/06_results_archive.")


if __name__ == "__main__":
    main()
