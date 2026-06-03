| model_name | calibrated_threshold | calibrated_balanced_accuracy | calibrated_precision | calibrated_recall | calibrated_f1 | recommended_threshold | recommended_balanced_accuracy | recommended_precision | recommended_recall | recommended_f1 | selection_rule | selected_for_reporting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| isolation_forest | 0.6131 | 0.5268 | 0.8968 | 0.6468 | 0.7515 | 0.6230 | 0.7453 | 0.9920 | 0.5245 | 0.6862 | max_balanced_accuracy_then_f1 | true |
| one_class_svm | 2.0919 | 0.5000 | 0.8885 | 1.0000 | 0.9409 | 5.4171 | 0.8724 | 0.9986 | 0.7532 | 0.8587 | max_balanced_accuracy_then_f1 | false |
