"""Re-run evaluation using previously saved models and train/test splits."""

from __future__ import annotations

import argparse
import json

from pdm_pipeline import ExperimentConfig, evaluate_saved_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate previously trained baseline models")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML file")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    summary = evaluate_saved_models(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
