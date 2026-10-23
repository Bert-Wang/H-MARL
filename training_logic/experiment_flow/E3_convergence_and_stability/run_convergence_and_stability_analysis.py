import os
import subprocess
import sys
from datetime import datetime


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_LOGIC_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
RESULTS_ROOT = os.path.join(TRAINING_LOGIC_ROOT, "06_results_archive")
RUNNER_LOG_DIR = os.path.join(RESULTS_ROOT, "experiment_runners", "E3_convergence_and_stability")
ANALYSIS_SCRIPT = os.path.join(
    TRAINING_LOGIC_ROOT,
    "05_analysis_tools",
    "utilities",
    "summarize_epoch_metrics.py",
)


def ensure_dirs():
    os.makedirs(RUNNER_LOG_DIR, exist_ok=True)


def run_analysis():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner_log_path = os.path.join(RUNNER_LOG_DIR, f"convergence_analysis_{timestamp}.log")

    print("[START] summarize_epoch_metrics.py")
    print(f"[SCRIPT] {ANALYSIS_SCRIPT}")
    print(f"[RUNNER LOG] {runner_log_path}")

    with open(runner_log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, ANALYSIS_SCRIPT],
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
        raise RuntimeError(
            f"Convergence analysis failed with exit code {return_code}. See {runner_log_path}"
        )

    print("[DONE] summarize_epoch_metrics.py")


def main():
    ensure_dirs()
    print("E3 Convergence and Stability analysis is starting.")
    print(f"Training logic root: {TRAINING_LOGIC_ROOT}")
    print(f"Results root: {RESULTS_ROOT}")
    run_analysis()
    print("E3 analysis finished successfully.")
    print("Plots: training_logic/06_results_archive/analysis/convergence_plots/")
    print("Tables: training_logic/06_results_archive/analysis/tables/")


if __name__ == "__main__":
    main()
