# Experiment Summary: kaist_rtf_experiment

## Problem Setting
- Dataset: `kaist_run_to_failure`
- Analysis type: anomaly-first chronological degradation analysis
- Signals used: vibration x/y, bearing temperature, ambient temperature
- Supervised classification: intentionally not used

## Data Summary
- Feature dataset: `data\processed\kaist_run_to_failure\datasets\kaist_rtf_feature_dataset.csv`
- Source files discovered: 129
- Sessions in feature dataset: 1
- Hourly feature rows: 129
- Model features used: 15
- All-missing features dropped before calibration: 51

## Calibration Region
- Healthy reference fraction: 0.2
- Healthy reference max hours: 24.0
- Minimum reference files: 12
- Rolling window size: 5 files
- Warning ratio threshold: 0.6
- kaist_rtf_vibration_bearing_runtofailure: 24 calibration files from hour 0.0 to 23.0 using `first_24_hours`

## Model Outcomes
- isolation_forest: threshold=0.711477, first_alarm_hour=nan, reason=no_alarm, status=no_alarm_detected
- one_class_svm: threshold=0.022606, first_alarm_hour=26.0, reason=sustained_warning, status=alarm_detected

## Conservative Interpretation
- The calibration region is used for model fitting and threshold calibration only.
- Later threshold crossings indicate deviation from the early run behavior, not a proven failure prediction timestamp.
- Because explicit failure-onset labels are unavailable, the results should be interpreted as anomaly-trend evidence rather than validated RUL or fault-time estimation.
