"""Configuration objects for the thesis-oriented baseline pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class PathsConfig:
    """Filesystem locations used by the experiment."""

    data_root: str = "data/example"
    processed_root: str = "data/processed"
    results_root: str = "results"
    feature_dataset_path: Optional[str] = None
    inference_session_dir: Optional[str] = None
    session_whitelist: List[str] = field(default_factory=list)


@dataclass
class DatasetConfig:
    """Dataset selection and schema choices for multi-dataset experiments."""

    name: str = "session_folder_baseline"
    variant: str = "default"
    group_column: str = "session_id"
    label_column: str = "label"
    multiclass_label_column: str = "multiclass_label"


@dataclass
class RunToFailureConfig:
    """Settings for compact run-to-failure adaptation and anomaly analysis."""

    sampling_rate_hz: float = 25600.0
    healthy_reference_fraction: float = 0.2
    healthy_reference_max_hours: Optional[float] = None
    minimum_reference_files: int = 12
    rolling_window_files: int = 5


@dataclass
class WindowConfig:
    """Windowing parameters shared by all sensor modalities."""

    duration_sec: float = 2.0
    overlap: float = 0.5
    minimum_samples: Dict[str, int] = field(
        default_factory=lambda: {
            "ae": 16,
            "vibration": 16,
            "thermal": 2,
        }
    )


@dataclass
class PreprocessingConfig:
    """Basic data cleaning parameters."""

    interpolation_method: str = "linear"
    fill_missing: bool = True


@dataclass
class SynchronizationConfig:
    """Synchronization strategy for asynchronous sensor streams."""

    method: str = "trim_to_overlap"


@dataclass
class ClassifierConfig:
    """Random Forest settings for health-state classification."""

    n_estimators: int = 300
    random_state: int = 42
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    n_jobs: int = -1
    class_weight: str = "balanced"
    balance_strategy: str = "none"
    balance_target_ratio: float = 1.0


@dataclass
class AnomalyConfig:
    """Isolation Forest settings for early anomaly warning."""

    n_estimators: int = 200
    random_state: int = 42
    contamination: str = "auto"
    one_class_svm_nu: float = 0.05
    one_class_svm_kernel: str = "rbf"
    one_class_svm_gamma: str = "scale"
    threshold_strategy: str = "quantile"
    threshold_quantile: float = 0.99
    threshold_buffer_std: float = 3.0
    minimum_healthy_windows: int = 10
    threshold_sweep_points: int = 101


@dataclass
class ModelConfig:
    """Container for the baseline models."""

    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)


@dataclass
class EvaluationConfig:
    """Evaluation choices used during train/test assessment."""

    healthy_label: str = "healthy"
    test_fraction: float = 0.25
    random_state: int = 42
    split_strategy: str = "session"
    positive_labels: List[str] = field(default_factory=list)


@dataclass
class PlotConfig:
    """Simple plotting controls for thesis-friendly figures."""

    top_n_feature_trends: int = 4


@dataclass
class InferenceConfig:
    """Thresholds used during session-level inference."""

    warning_ratio_threshold: float = 0.2


@dataclass
class AblationConfig:
    """Configuration for single-sensor versus fusion comparisons."""

    enabled: bool = True
    include_metadata: bool = False
    include_fusion_features: bool = True
    setups: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "ae_only": ["ae"],
            "vibration_only": ["vibration"],
            "thermal_only": ["thermal"],
            "fused": ["ae", "vibration", "thermal"],
        }
    )


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    experiment_name: str = "thesis_baseline"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    run_to_failure: RunToFailureConfig = field(default_factory=RunToFailureConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    synchronization: SynchronizationConfig = field(default_factory=SynchronizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load a configuration file while keeping defaults for omitted fields."""

        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        return cls(
            experiment_name=raw.get("experiment_name", cls().experiment_name),
            dataset=DatasetConfig(**raw.get("dataset", {})),
            run_to_failure=RunToFailureConfig(**raw.get("run_to_failure", {})),
            paths=PathsConfig(**raw.get("paths", {})),
            window=WindowConfig(**raw.get("window", {})),
            preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
            synchronization=SynchronizationConfig(**raw.get("synchronization", {})),
            model=ModelConfig(
                classifier=ClassifierConfig(**raw.get("model", {}).get("classifier", {})),
                anomaly=AnomalyConfig(**raw.get("model", {}).get("anomaly", {})),
            ),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
            plots=PlotConfig(**raw.get("plots", {})),
            inference=InferenceConfig(**raw.get("inference", {})),
            ablation=AblationConfig(**raw.get("ablation", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary that can be saved as YAML or JSON."""

        return asdict(self)
