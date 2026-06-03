"""Build the compact Paderborn dataset artifacts."""

from __future__ import annotations

import argparse
import json

from pdm_pipeline.config import ExperimentConfig
from pdm_pipeline.paderborn_adapter import build_paderborn_compact_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the compact Paderborn feature dataset.")
    parser.add_argument("--config", required=True, help="Path to the Paderborn YAML file")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    summary = build_paderborn_compact_dataset(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
