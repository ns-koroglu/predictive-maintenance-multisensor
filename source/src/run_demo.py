"""Run the full presentation demo in one command."""

from __future__ import annotations

import argparse
import json

from pdm_pipeline import ExperimentConfig, run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full presentation-grade predictive maintenance demo")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML file")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    summary = run_demo(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
