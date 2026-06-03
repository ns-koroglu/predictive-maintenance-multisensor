"""Build the fused dataset using the config-driven baseline package."""

from __future__ import annotations

import argparse
import json

from pdm_pipeline import ExperimentConfig, build_dataset_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fused multi-sensor dataset from session folders")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML file")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    summary = build_dataset_from_config(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
