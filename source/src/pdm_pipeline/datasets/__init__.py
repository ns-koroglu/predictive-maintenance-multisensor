"""Dataset registry and standardized schema helpers."""

from .registry import get_dataset_metadata, list_supported_datasets
from .schema import (
    STANDARD_IDENTIFIER_COLUMNS,
    STANDARD_MODALITIES,
    STANDARD_MODALITY_FLAG_COLUMNS,
    DatasetMetadata,
    build_schema_overview,
    standardize_feature_dataset,
    summarize_modality_availability,
)

__all__ = [
    "DatasetMetadata",
    "STANDARD_IDENTIFIER_COLUMNS",
    "STANDARD_MODALITIES",
    "STANDARD_MODALITY_FLAG_COLUMNS",
    "build_schema_overview",
    "get_dataset_metadata",
    "list_supported_datasets",
    "standardize_feature_dataset",
    "summarize_modality_availability",
]
