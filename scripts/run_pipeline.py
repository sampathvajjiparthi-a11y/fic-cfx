"""Run the whole experiment for one dataset, step by step.

    python scripts/run_pipeline.py --dataset small --quick        # 2-5 minute smoke test
    python scripts/run_pipeline.py --dataset small                # full run (hours on CPU)
    python scripts/run_pipeline.py --dataset small --skip resnet18
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script: str, *extra: str) -> None:
    cmd = [sys.executable, str(HERE / script), *extra]
    print("\n>>>", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="small")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    p.add_argument("--skip", nargs="*", default=[], help="models to skip, e.g. resnet18")
    a = p.parse_args()
    common = ["--dataset", a.dataset] + (["--quick"] if a.quick else [])
    if a.config:
        common += ["--config", a.config]
    classical = [m for m in ("logistic_regression", "random_forest") if m not in a.skip]
    deep = [m for m in ("cnn", "resnet18") if m not in a.skip]
    models = classical + deep

    run("01_check_data.py", *common)
    if classical:
        run("02_train_classical.py", *common, "--models", *classical)
    for m in deep:
        run("03_train_deep.py", *common, "--model", m)
    run("04_train_autoencoder.py", *common)
    run("05_evaluate_models.py", *common, "--models", *models)
    run("06_explain_xai.py", *common, "--models", *models)
    for m in models:
        methods = ["blackbox"] if m == "random_forest" else ["wachter", "prototype"]
        for k in methods:
            run("07_generate_counterfactuals.py", *common, "--model", m, "--method", k,
                "--select_from", *models)
    run("08_analyse_counterfactuals.py", *common)
    print("\nDone. Results are in outputs/metrics, outputs/figures and outputs/counterfactuals.")


if __name__ == "__main__":
    main()
