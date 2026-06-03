"""Train the baseline classifier and anomaly detector from one YAML config."""

from __future__ import annotations

import argparse
import json

from pdm_pipeline import ExperimentConfig, run_training_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline predictive maintenance models")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML file")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    summary = run_training_experiment(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
