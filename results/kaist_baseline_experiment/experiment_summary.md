# Experiment Summary: kaist_baseline_experiment

## Input
- Feature dataset: `results\kaist_feature_build\datasets\kaist_vibration_thermal_features.csv`
- Baseline modalities: `vibration + thermal`
- Current: exported and preserved in compact KAIST metadata, but not used in this first baseline
- Session semantics: condition-matched and explicitly unsynchronized

## Split
- Split strategy: `session`
- Train windows: 3352
- Test windows: 1058
- Train sessions: kaist_0nm_bpfi_03, kaist_0nm_bpfi_10, kaist_0nm_bpfi_30, kaist_0nm_bpfo_03, kaist_0nm_bpfo_30, kaist_0nm_misalignment_01, kaist_0nm_misalignment_05, kaist_0nm_normal, kaist_0nm_unbalance_0583mg, kaist_0nm_unbalance_1169mg, kaist_0nm_unbalance_1751mg, kaist_0nm_unbalance_2239mg, kaist_0nm_unbalance_3318mg, kaist_2nm_bpfi_03, kaist_2nm_bpfi_10, kaist_2nm_bpfi_30, kaist_2nm_bpfo_03, kaist_2nm_bpfo_30, kaist_2nm_misalignment_01, kaist_2nm_misalignment_05, kaist_2nm_unbalance_0583mg, kaist_2nm_unbalance_1169mg, kaist_2nm_unbalance_3318mg, kaist_4nm_bpfi_30, kaist_4nm_bpfo_03, kaist_4nm_bpfo_10, kaist_4nm_bpfo_30, kaist_4nm_misalignment_01, kaist_4nm_misalignment_03, kaist_4nm_misalignment_05, kaist_4nm_normal, kaist_4nm_unbalance_0583mg, kaist_4nm_unbalance_1169mg, kaist_4nm_unbalance_2239mg
- Test sessions: kaist_0nm_bpfo_10, kaist_0nm_misalignment_03, kaist_2nm_bpfo_10, kaist_2nm_misalignment_03, kaist_2nm_normal, kaist_2nm_unbalance_1751mg, kaist_2nm_unbalance_2239mg, kaist_4nm_bpfi_03, kaist_4nm_bpfi_10, kaist_4nm_unbalance_1751mg, kaist_4nm_unbalance_3318mg
- Session overlap: none

## Class Imbalance
- Train label counts before balancing: {'faulty': 2936, 'healthy': 416}
- Train label counts used by the classifier: {'healthy': 416, 'faulty': 416}
- Train-only balancing strategy: `downsample_majority`
- Majority-class test accuracy baseline: 0.8885
- Raw accuracy is misleading here because most KAIST windows are faulty. A classifier can look strong by predicting only `faulty`, while still achieving zero healthy-class recall.

## Classification
- Accuracy: 0.888468809073724
- Balanced accuracy: 0.5
- Macro precision: 0.444234404536862
- Macro recall: 0.5
- Macro F1: 0.47047047047047047
- ROC-AUC: 0.39555084745762714
- PR-AUC: 0.8655214322872207
- Predicted label counts on the test set: {'faulty': 1058}

## Anomaly Baselines
- isolation_forest: balanced_accuracy=0.5268, precision=0.8968, recall=0.6468, f1=0.7515, threshold=0.613057, recommended_threshold=0.622958
- one_class_svm: balanced_accuracy=0.5000, precision=0.8885, recall=1.0000, f1=0.9409, threshold=2.091869, recommended_threshold=5.417093

## Selected Anomaly Baseline
- Default reported anomaly baseline: `isolation_forest`
- Isolation Forest remains the default reported anomaly baseline for conservative interpretation.
- Calibrated threshold: 0.6130571930759808
- Recommended threshold from the sweep: 0.6229575929642762
- The recommended threshold is an evaluation-side analysis aid from the held-out sweep. The calibrated threshold remains the train-only operating point.

## Split Sensitivity
- Classifier balanced accuracy range across healthy holdout splits: 0.4851 to 0.5000
- Isolation Forest calibrated balanced accuracy range: 0.3782 to 0.4340
- Isolation Forest sweep-optimized balanced accuracy range: 0.5368 to 0.6569
