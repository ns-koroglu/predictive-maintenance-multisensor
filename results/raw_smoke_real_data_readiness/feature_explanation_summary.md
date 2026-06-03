# Feature Explanation Summary

This explanation uses the fused Random Forest feature importances from the baseline classifier.

## Most Important Features
- `vibration_az_max`: importance 0.0333
- `vibration_ax_spec_centroid`: importance 0.0300
- `thermal_t_max_max`: importance 0.0233
- `vibration_ay_rms`: importance 0.0233
- `vibration_ay_std`: importance 0.0233
- `fusion_thermal_to_ae_ratio`: importance 0.0233
- `thermal_t_max_start`: importance 0.0233
- `thermal_temp_gap_max`: importance 0.0233
- `vibration_ay_kurtosis`: importance 0.0200
- `vibration_az_min`: importance 0.0200

## Sensor Group Contribution
- Dominant sensor group: `vibration`
- Normalized importance share: 0.590
- Excluded non-sensor importance (fusion/meta/context): 0.0900

## Interpretation
The strongest modality-specific contribution came from the vibration feature family with normalized importance 0.590. The most important fused-model features were vibration_az_max, vibration_ax_spec_centroid, thermal_t_max_max. This supports the fusion rationale because the final model still depends on multiple sensor-derived features rather than a single modality alone.
