"""Run a configured experiment through the unified dataset-aware framework."""

from __future__ import annotations

import argparse
import json

from pdm_pipeline import ExperimentConfig, run_configured_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a dataset-aware predictive maintenance experiment")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML file")
    parser.add_argument(
        "--stage",
        default="train",
        choices=["build", "train", "evaluate", "infer", "demo"],
        help="Experiment stage to run",
    )
    parser.add_argument("--session-dir", help="Optional session override used only during inference")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    summary = run_configured_experiment(config, stage=args.stage, session_dir=args.session_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
