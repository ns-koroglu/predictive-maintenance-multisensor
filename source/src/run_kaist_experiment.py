"""Run the KAIST baseline experiment from the compact feature dataset."""

from __future__ import annotations

import argparse
import json

from pdm_pipeline.config import ExperimentConfig
from pdm_pipeline.kaist_experiment import run_kaist_baseline_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the KAIST baseline experiment.")
    parser.add_argument("--config", required=True, help="Path to the KAIST experiment YAML file")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    summary = run_kaist_baseline_experiment(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
