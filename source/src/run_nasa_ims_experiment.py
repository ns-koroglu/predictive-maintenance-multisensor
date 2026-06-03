"""Run the compact anomaly-first NASA IMS experiment."""

from __future__ import annotations

import argparse
import json

from pdm_pipeline.config import ExperimentConfig
from pdm_pipeline.nasa_ims_experiment import run_nasa_ims_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NASA IMS anomaly-first experiment.")
    parser.add_argument("--config", required=True, help="Path to the NASA IMS YAML file")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    summary = run_nasa_ims_experiment(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
