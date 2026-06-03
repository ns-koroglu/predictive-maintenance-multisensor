"""Run the binary vibration-only Paderborn baseline experiment."""

from __future__ import annotations

import argparse
import json

from pdm_pipeline.config import ExperimentConfig
from pdm_pipeline.paderborn_experiment import run_paderborn_baseline_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Paderborn binary vibration-only benchmark.")
    parser.add_argument("--config", required=True, help="Path to the Paderborn YAML file")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    summary = run_paderborn_baseline_experiment(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
