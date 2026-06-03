"""Static registry of datasets supported by the research framework."""

from __future__ import annotations

from typing import Dict, List

from .schema import DatasetMetadata


DATASET_REGISTRY: Dict[str, DatasetMetadata] = {
    "session_folder_baseline": DatasetMetadata(
        key="session_folder_baseline",
        display_name="Session Folder Baseline",
        implementation_status="implemented",
        default_input_kind="session_folders",
        supported_modalities=("ae", "vibration", "thermal"),
        recommended_baseline_modalities=("ae", "vibration", "thermal"),
        notes="Existing repository baseline using one folder per session.",
        publication_caution="Use session-aware evaluation to avoid window leakage.",
    ),
    "kaist_rotating_machine": DatasetMetadata(
        key="kaist_rotating_machine",
        display_name="KAIST Rotating Machine",
        implementation_status="implemented",
        default_input_kind="feature_dataset",
        supported_modalities=("vibration", "thermal", "current", "acoustic"),
        recommended_baseline_modalities=("vibration", "thermal"),
        notes="Compact KAIST path currently uses vibration + thermal features for the first baseline.",
        publication_caution="Sessions are condition-matched and explicitly unsynchronized.",
    ),
    "kaist_run_to_failure": DatasetMetadata(
        key="kaist_run_to_failure",
        display_name="KAIST Run-to-Failure",
        implementation_status="implemented",
        default_input_kind="hourly_measurement_files",
        supported_modalities=("vibration", "thermal"),
        recommended_baseline_modalities=("vibration", "thermal"),
        notes="Compact anomaly-first path using hourly vibration and temperature files.",
        publication_caution=(
            "Use chronological reference calibration and keep run-level progression explicit. "
            "Do not report supervised metrics unless explicit labels are available."
        ),
    ),
    "nasa_ims": DatasetMetadata(
        key="nasa_ims",
        display_name="NASA IMS",
        implementation_status="implemented",
        default_input_kind="bearing_snapshot_files",
        group_column="group_id",
        supported_modalities=("vibration",),
        recommended_baseline_modalities=("vibration",),
        notes=(
            "Compact anomaly-first path using timestamp-named ASCII vibration snapshots. "
            "One bearing trajectory is one session; one test run is one group."
        ),
        publication_caution=(
            "Chronology must come from filename timestamps only. "
            "Keep documented end-of-run failures separate from dense row-level labels."
        ),
    ),
    "paderborn": DatasetMetadata(
        key="paderborn",
        display_name="Paderborn",
        implementation_status="implemented",
        default_input_kind="compact_multirate_snapshot_files",
        group_column="group_id",
        supported_modalities=("vibration", "current", "thermal"),
        recommended_baseline_modalities=("vibration",),
        notes=(
            "Compact snapshot path using one .mat measurement file as one feature row. "
            "The first benchmark is binary healthy vs faulty with vibration-only features."
        ),
        publication_caution=(
            "Use bearing-code group-aware splitting so the same physical bearing never leaks across train/test. "
            "Keep multiclass semantics conservative and PDF-derived."
        ),
    ),
    "cwru": DatasetMetadata(
        key="cwru",
        display_name="CWRU",
        implementation_status="scaffold_only",
        default_input_kind="adapter_pending",
        notes="Registry entry only. Adapter and feature builder are not implemented yet.",
        publication_caution="Avoid file-level leakage across operating conditions and fault severities.",
    ),
}


def list_supported_datasets() -> List[DatasetMetadata]:
    """Return the registered dataset entries in a stable order."""

    return [DATASET_REGISTRY[key] for key in sorted(DATASET_REGISTRY)]


def get_dataset_metadata(dataset_name: str) -> DatasetMetadata:
    """Return one dataset entry or raise a clear error."""

    normalized_name = str(dataset_name)
    if normalized_name not in DATASET_REGISTRY:
        supported = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unsupported dataset '{normalized_name}'. Available datasets: {supported}")
    return DATASET_REGISTRY[normalized_name]
