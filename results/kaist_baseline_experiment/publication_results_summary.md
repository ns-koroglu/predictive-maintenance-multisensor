# Publication-Ready Results Summary

## Problem Setting
This analysis evaluates a thesis-oriented predictive maintenance baseline on the compact KAIST rotating machine feature dataset.
The first integration baseline uses vibration and thermal features only. Sessions are condition-matched rather than safely time-synchronous, so all evaluation remains session-aware and leakage-safe.

## Class Imbalance
The dataset contains 45 sessions and a binary label distribution of {'faulty': 3876, 'healthy': 534}.
Only three sessions are healthy, which makes generalization to unseen healthy conditions difficult and makes any window-level accuracy figure potentially deceptive.

## Why Raw Accuracy Is Misleading
The supervised classifier reached an apparent accuracy of 0.8885.
However, this is nearly identical to the majority-class baseline because the test set is dominated by faulty windows.
The more informative metrics are balanced accuracy (0.5000) and macro F1 (0.4705), which show that the classifier does not generalize to the healthy class.

## Classifier Failure Case
Under the current session-safe split, the binary Random Forest predicts every test window as faulty.
This yields zero healthy-class recall despite train-time downsampling of the faulty class.
The supervised result should therefore be interpreted as a failure case under extreme class imbalance, not as a strong baseline.

## Anomaly-First Interpretation
Healthy-only anomaly detection is more informative in this setting because it matches the small-healthy-data constraint more naturally.
The default reported anomaly baseline remains Isolation Forest for conservative reporting.
Its calibrated operating point gives balanced accuracy 0.5268, precision 0.8968, recall 0.6468, and F1 0.7515.
One-Class SVM is also reported for comparison, but it is not used as the default headline result because its calibration is more sensitive to the tiny healthy training set.

## Threshold Calibration Interpretation
For the default Isolation Forest baseline, the train-only calibrated threshold is 0.613057.
The held-out sweep identifies a more favorable analysis threshold of 0.622958, with higher balanced accuracy 0.7453.
This sweep-selected threshold is intentionally reported as optimistic analysis only. It must not be conflated with the train-only calibrated operating point.

## Split-Sensitivity Analysis
A small split-sensitivity study rotates the held-out healthy session across all three healthy sessions while preserving session-safe splitting.
Across these splits, classifier balanced accuracy ranges from 0.4851 to 0.5000.
Isolation Forest calibrated balanced accuracy ranges from 0.3782 to 0.4340, while the optimistic sweep-selected value ranges from 0.5368 to 0.6569.
This reinforces the same conclusion: threshold choice matters, and any anomaly headline result should be interpreted conservatively.

## Conservative Conclusion
The current KAIST baseline does not support a reliable supervised healthy-vs-faulty classifier under the present session-safe split and class distribution.
The anomaly-first view is more defensible, especially when reported with explicit separation between calibrated operating points and optimistic sweep-based analysis.
