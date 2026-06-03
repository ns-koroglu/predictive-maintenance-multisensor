"""Adapt the KAIST rotating machine dataset into the repository session schema."""

from __future__ import annotations

import argparse
import json

from pdm_pipeline.kaist_adapter import adapt_kaist_dataset, run_kaist_compact_workflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize and export the KAIST rotating machine dataset."
    )
    parser.add_argument(
        "--mode",
        choices=["compact", "full_raw"],
        default="compact",
        help="Compact mode is the storage-efficient default. Full raw mode writes raw-sample CSV exports.",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/external/kaist_rotating_machine/extracted",
        help="Path to the extracted KAIST dataset root.",
    )
    parser.add_argument(
        "--processed-root",
        default="data/processed/kaist_rotating_machine",
        help="Output directory for processed session-level exports.",
    )
    parser.add_argument(
        "--interim-root",
        default="data/interim/kaist_rotating_machine",
        help="Used only in full_raw mode for large interim modality-level exports.",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Results root used in compact mode for direct feature builds.",
    )
    parser.add_argument(
        "--experiment-name",
        default="kaist_feature_build",
        help="Result folder name used in compact mode.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=128,
        help="Number of preview rows saved per modality in compact mode.",
    )
    parser.add_argument(
        "--window-duration-sec",
        type=float,
        default=2.0,
        help="Window duration used for direct feature building in compact mode.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Window overlap fraction used for direct feature building in compact mode.",
    )
    args = parser.parse_args()

    if args.mode == "compact":
        summary = run_kaist_compact_workflow(
            dataset_root=args.dataset_root,
            processed_root=args.processed_root,
            results_root=args.results_root,
            experiment_name=args.experiment_name,
            preview_rows=args.preview_rows,
            window_duration_sec=args.window_duration_sec,
            overlap=args.overlap,
        )
    else:
        summary = adapt_kaist_dataset(
            dataset_root=args.dataset_root,
            interim_root=args.interim_root,
            processed_root=args.processed_root,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
