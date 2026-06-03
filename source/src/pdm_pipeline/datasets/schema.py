"""Dataset metadata and standardized experiment-schema helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd


STANDARD_MODALITIES: tuple[str, ...] = ("ae", "acoustic", "vibration", "thermal", "current")
STANDARD_MODALITY_FLAG_COLUMNS: tuple[str, ...] = tuple(f"has_{name}" for name in STANDARD_MODALITIES)
STANDARD_IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "dataset_name",
    "dataset_variant",
    "dataset_display_name",
    "session_id",
    "group_id",
    "label",
    "multiclass_label",
    "reference_region_role",
    "window_index",
    "window_start",
    "window_end",
    "split_group",
    "window_pairing_strategy",
)


@dataclass(frozen=True)
class DatasetMetadata:
    """Static registry entry describing one supported dataset family."""

    key: str
    display_name: str
    implementation_status: str
    default_input_kind: str
    group_column: str = "session_id"
    label_column: str = "label"
    multiclass_label_column: str = "multiclass_label"
    supported_modalities: tuple[str, ...] = field(default_factory=tuple)
    recommended_baseline_modalities: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""
    publication_caution: str = ""

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-safe dictionary for reporting."""

        return asdict(self)


def infer_modality_flags_from_columns(feature_frame: pd.DataFrame) -> Dict[str, bool]:
    """Infer modality availability from standard feature prefixes."""

    prefixes = {
        "ae": ("ae_",),
        "acoustic": ("acoustic_",),
        "vibration": ("vibration_",),
        "thermal": ("thermal_",),
        "current": ("current_",),
    }
    flags: Dict[str, bool] = {}
    for modality_name, candidates in prefixes.items():
        flags[modality_name] = any(
            str(column).startswith(prefix) for prefix in candidates for column in feature_frame.columns
        )
    return flags


def build_default_modality_flags(
    metadata: DatasetMetadata,
    feature_frame: pd.DataFrame,
    overrides: Mapping[str, bool] | None = None,
) -> Dict[str, bool]:
    """Resolve a complete modality-availability mapping for a feature dataset."""

    inferred = infer_modality_flags_from_columns(feature_frame)
    flags = {name: bool(inferred.get(name, False)) for name in STANDARD_MODALITIES}
    for modality_name in metadata.supported_modalities:
        flags[str(modality_name)] = bool(flags.get(str(modality_name), False))
    if overrides:
        for key, value in overrides.items():
            if key in STANDARD_MODALITIES:
                flags[key] = bool(value)
    return flags


def standardize_feature_dataset(
    frame: pd.DataFrame,
    metadata: DatasetMetadata,
    dataset_variant: str = "default",
    group_column: str | None = None,
    label_column: str | None = None,
    multiclass_label_column: str | None = None,
    modality_flags: Mapping[str, bool] | None = None,
) -> pd.DataFrame:
    """Apply a shared internal experiment schema to a window-level feature dataset."""

    resolved_group_column = group_column or metadata.group_column
    resolved_label_column = label_column or metadata.label_column
    resolved_multiclass_column = multiclass_label_column or metadata.multiclass_label_column

    missing = [name for name in (resolved_group_column, resolved_label_column) if name not in frame.columns]
    if missing:
        raise ValueError(
            "Standardized feature datasets require the configured group and label columns. "
            f"Missing columns: {missing}"
        )

    standardized = frame.copy()
    if "session_id" in standardized.columns:
        standardized["session_id"] = standardized["session_id"].astype(str)
    elif resolved_group_column in standardized.columns:
        standardized["session_id"] = standardized[resolved_group_column].astype(str)
    else:
        raise ValueError("Standardized feature datasets require a 'session_id' column or a valid fallback group column.")

    if resolved_group_column in standardized.columns:
        standardized["group_id"] = standardized[resolved_group_column].astype(str)
    elif "group_id" in standardized.columns:
        standardized["group_id"] = standardized["group_id"].astype(str)
    else:
        standardized["group_id"] = standardized["session_id"].astype(str)
    standardized["split_group"] = standardized["group_id"]

    if resolved_label_column != "label":
        standardized["label"] = standardized[resolved_label_column].astype(str)
    else:
        standardized["label"] = standardized["label"].astype(str)

    if resolved_multiclass_column in standardized.columns:
        if resolved_multiclass_column != "multiclass_label":
            standardized["multiclass_label"] = standardized[resolved_multiclass_column].astype(str)
        else:
            standardized["multiclass_label"] = standardized["multiclass_label"].astype(str)
    else:
        standardized["multiclass_label"] = standardized["label"].astype(str)

    standardized["dataset_name"] = metadata.key
    standardized["dataset_variant"] = str(dataset_variant)
    standardized["dataset_display_name"] = metadata.display_name

    if "window_index" not in standardized.columns:
        standardized["window_index"] = standardized.groupby("session_id").cumcount().astype(int)
    if "window_start" not in standardized.columns:
        standardized["window_start"] = np.nan
    if "window_end" not in standardized.columns:
        standardized["window_end"] = np.nan
    if "window_pairing_strategy" not in standardized.columns:
        standardized["window_pairing_strategy"] = "dataset_defined"
    if "reference_region_role" not in standardized.columns:
        standardized["reference_region_role"] = "unknown"

    flags = build_default_modality_flags(metadata, standardized, overrides=modality_flags)
    for modality_name in STANDARD_MODALITIES:
        standardized[f"has_{modality_name}"] = bool(flags.get(modality_name, False))

    return standardized


def summarize_modality_availability(frame: pd.DataFrame) -> Dict[str, int]:
    """Summarize how many rows declare each modality as available."""

    summary: Dict[str, int] = {}
    for column_name in STANDARD_MODALITY_FLAG_COLUMNS:
        if column_name in frame.columns:
            summary[column_name] = int(frame[column_name].astype(bool).sum())
    return summary


def build_schema_overview(metadata: DatasetMetadata) -> Dict[str, object]:
    """Return a compact schema description for reports and registry listings."""

    return {
        "dataset_name": metadata.key,
        "group_column": metadata.group_column,
        "label_column": metadata.label_column,
        "multiclass_label_column": metadata.multiclass_label_column,
        "standard_identifier_columns": list(STANDARD_IDENTIFIER_COLUMNS),
        "standard_modality_flag_columns": list(STANDARD_MODALITY_FLAG_COLUMNS),
        "supported_modalities": list(metadata.supported_modalities),
        "recommended_baseline_modalities": list(metadata.recommended_baseline_modalities),
    }
