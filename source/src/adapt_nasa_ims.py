"""Build compact manifests and features for the NASA IMS bearing dataset."""

from __future__ import annotations

import argparse
import json

from pdm_pipeline.config import ExperimentConfig
from pdm_pipeline.nasa_ims_adapter import build_nasa_ims_compact_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the compact NASA IMS adapter outputs.")
    parser.add_argument("--config", required=True, help="Path to the NASA IMS YAML file")
    parser.add_argument("--debug", action="store_true", help="Process only the first few snapshot files")
    parser.add_argument("--max-files", type=int, help="Optional hard limit for the number of snapshot files")
    parser.add_argument("--progress-every", type=int, default=250, help="Print progress every N processed files")
    parser.add_argument("--quiet", action="store_true", help="Reduce progress logging")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    summary = build_nasa_ims_compact_dataset(
        config,
        debug=bool(args.debug),
        max_files=args.max_files,
        progress_every=int(args.progress_every),
        verbose=not bool(args.quiet),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
