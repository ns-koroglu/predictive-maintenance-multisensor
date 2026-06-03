# Demo Summary: raw_smoke_real_data_readiness

## Sessions Used
- Data root: `data/raw`
- Session whitelist: session_001_healthy, session_002_healthy, session_003_developing_fault, session_004_developing_fault
- Generated sessions: 4

## Class Distribution
- Window counts: {'healthy': 22, 'developing_fault': 22}

## Train/Test Split
- Split strategy: `session`
- Train sessions: session_002_healthy, session_003_developing_fault
- Test sessions: session_001_healthy, session_004_developing_fault

## Key Metrics
- Classifier macro precision: 1.0
- Classifier macro recall: 1.0
- Classifier macro F1: 1.0
- Classifier ROC-AUC: 1.0
- Anomaly precision: 1.0
- Anomaly recall: 0.9090909090909091
- Anomaly F1: 0.9523809523809523
- Anomaly threshold: 0.5688793488131487

## Inference Result
- Session: `session_004_developing_fault`
- Predicted class: `developing_fault`
- Early warning: `True`
- Warning trigger reason: `warning triggered by both`
- Decision summary: Predicted class is 'developing_fault', anomaly ratio is 0.909, and the anomaly warning threshold is 0.200. The final decision is: warning triggered by both.

## Why Fusion?
Fusion gave the strongest overall result in this demo. Its mean F1 across classification and anomaly detection was 0.950, compared with 0.500 for the best single sensor setup (ae_only).

## Which Sensor Signals Contributed Most?
The strongest modality-specific contribution came from the vibration feature family with normalized importance 0.590. The most important fused-model features were vibration_az_max, vibration_ax_spec_centroid, thermal_t_max_max. This supports the fusion rationale because the final model still depends on multiple sensor-derived features rather than a single modality alone.

## Per-Session Results
| session_id | true_label | split | demo_role | n_windows | majority_predicted_label | window_accuracy | mean_max_probability | mean_anomaly_score | anomalous_window_ratio | inference_predicted_class | inference_early_warning | warning_trigger_reason | decision_summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| session_001_healthy | healthy | test | dataset_session | 11 | healthy | 1.0 | 0.9554545454545454 | 0.5009828575700888 | 0.0 |  |  |  |  |
| session_002_healthy | healthy | train | dataset_session | 11 | nan | nan | nan | nan | nan |  |  |  |  |
| session_003_developing_fault | developing_fault | train | dataset_session | 11 | nan | nan | nan | nan | nan |  |  |  |  |
| session_004_developing_fault | developing_fault | test | inference_target | 11 | developing_fault | 1.0 | 0.9736363636363636 | 0.5764981385576133 | 0.9090909090909092 | developing_fault | True | warning triggered by both | Predicted class is 'developing_fault', anomaly ratio is 0.909, and the anomaly warning threshold is 0.200. The final decision is: warning triggered by both. |

## Ablation Outputs
- `ablation_comparison.csv`
- `ablation_comparison.json`
- `ablation_summary.md`
- `plots/ablation_comparison.png`

## Explanation Outputs
- `top_features.csv`
- `top_features.json`
- `feature_explanation_summary.md`
- `sensor_group_importance.csv`
- `sensor_group_importance.json`
- `plots/top_features.png`
- `plots/sensor_group_importance.png`

## Presentation Figures
- `plots/class_distribution.png`
- `plots/confusion_matrix.png`
- `plots/anomaly_scores.png`
- `plots/feature_importances.png`
