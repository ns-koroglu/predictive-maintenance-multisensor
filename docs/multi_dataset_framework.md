# Multi-Dataset Framework Scaffold

## Purpose

This repository now includes a lightweight multi-dataset framework layer that sits above the existing working baselines.

The framework does not replace the current pipelines. It provides:

- a dataset registry
- a shared internal experiment schema
- config-driven dataset selection
- a unified runner entry point
- explicit modality-availability fields
- preservation of group-aware and session-aware evaluation

## Registered Datasets

- `session_folder_baseline`
- `kaist_rotating_machine`
- `kaist_run_to_failure`
- `nasa_ims`
- `paderborn`
- `cwru`

Only these dataset paths are currently implemented:

- `session_folder_baseline`
- `kaist_rotating_machine`
- `kaist_run_to_failure`
- `nasa_ims`

The other public datasets are registered as scaffold entries only. This keeps the framework explicit and ready for later adapter work without fabricating support prematurely.

## Shared Internal Experiment Schema

The framework standardizes window-level feature datasets to include:

- `dataset_name`
- `dataset_variant`
- `dataset_display_name`
- `session_id`
- `group_id`
- `label`
- `multiclass_label`
- `window_index`
- `window_start`
- `window_end`
- `split_group`
- `window_pairing_strategy`

Explicit modality-availability columns are also added:

- `has_ae`
- `has_acoustic`
- `has_vibration`
- `has_thermal`
- `has_current`

These fields are metadata, not model features.

## Unified Runner

The new entry point is:

- `python src/run_experiment.py --config <config_path> --stage <stage>`

Supported stages:

- `build`
- `train`
- `evaluate`
- `infer`
- `demo`

Current dispatch behavior:

- `session_folder_baseline` routes to the existing baseline/demo pipeline
- `kaist_rotating_machine` routes to the current KAIST experiment runner
- `kaist_run_to_failure` routes to the compact anomaly-first run-to-failure pipeline
- `nasa_ims` routes to the compact anomaly-first bearing-trajectory pipeline
- scaffold-only datasets raise a clear `NotImplementedError`

## Design Constraint

This layer is intentionally minimal. It standardizes orchestration and metadata first, while leaving dataset-specific adapters and feature builders to later implementation work.
