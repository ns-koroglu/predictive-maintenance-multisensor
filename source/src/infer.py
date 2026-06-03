"""Run config-driven inference for one session directory."""

from __future__ import annotations

import argparse
import json

from pdm_pipeline import ExperimentConfig, run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for one multi-sensor session")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML file")
    parser.add_argument("--session-dir", help="Optional override for the session folder used at inference time")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    summary = run_inference(config, session_dir=args.session_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
