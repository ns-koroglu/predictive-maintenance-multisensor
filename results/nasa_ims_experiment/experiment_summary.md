# Experiment Summary: nasa_ims_experiment

## Problem Setting
- Dataset: `nasa_ims`
- Analysis type: anomaly-first bearing-level degradation analysis
- Signals used: vibration only
- Session semantics: one bearing trajectory per session, one test run per group
- Supervised classification: intentionally not used

## Data Summary
- Feature dataset: `data\processed\nasa_ims\datasets\nasa_ims_bearing_feature_dataset.csv`
- Discovered runs: 3
- Bearing sessions: 12
- Feature rows: 37856
- Model features used: 44
- All-missing features dropped before calibration: 0

## Calibration Region
- Healthy reference fraction: 0.2
- Healthy reference max hours: 24.0
- Minimum reference files: 12
- Rolling window size: 6 snapshots
- Warning ratio threshold: 0.6
- nasa_ims_1st_test_bearing_1: 156 calibration snapshots, hours 0.0 to 22.130277777777778, mode=`first_24_hours`
- nasa_ims_1st_test_bearing_2: 156 calibration snapshots, hours 0.0 to 22.130277777777778, mode=`first_24_hours`
- nasa_ims_1st_test_bearing_3: 156 calibration snapshots, hours 0.0 to 22.130277777777778, mode=`first_24_hours`
- nasa_ims_1st_test_bearing_4: 156 calibration snapshots, hours 0.0 to 22.130277777777778, mode=`first_24_hours`
- nasa_ims_2nd_test_bearing_1: 144 calibration snapshots, hours 0.0 to 23.83333333333333, mode=`first_24_hours`
- nasa_ims_2nd_test_bearing_2: 144 calibration snapshots, hours 0.0 to 23.83333333333333, mode=`first_24_hours`
- nasa_ims_2nd_test_bearing_3: 144 calibration snapshots, hours 0.0 to 23.83333333333333, mode=`first_24_hours`
- nasa_ims_2nd_test_bearing_4: 144 calibration snapshots, hours 0.0 to 23.83333333333333, mode=`first_24_hours`
- nasa_ims_3rd_test__4th_test__txt_bearing_1: 145 calibration snapshots, hours 0.0 to 23.916666666666668, mode=`first_24_hours`
- nasa_ims_3rd_test__4th_test__txt_bearing_2: 145 calibration snapshots, hours 0.0 to 23.916666666666668, mode=`first_24_hours`
- nasa_ims_3rd_test__4th_test__txt_bearing_3: 145 calibration snapshots, hours 0.0 to 23.916666666666668, mode=`first_24_hours`
- nasa_ims_3rd_test__4th_test__txt_bearing_4: 145 calibration snapshots, hours 0.0 to 23.916666666666668, mode=`first_24_hours`

## Model Outcomes
- isolation_forest: threshold=0.639746, reference_files=1780, sessions_with_alarm=4
- one_class_svm: threshold=2.244755, reference_files=1780, sessions_with_alarm=12

## First Alarm Behavior
- isolation_forest / nasa_ims_1st_test_bearing_1: first_alarm_hour=287.7555555555556, reason=threshold_crossing, documented_failure_mode=nan
- isolation_forest / nasa_ims_1st_test_bearing_2: first_alarm_hour=820.0188888888889, reason=sustained_warning, documented_failure_mode=nan
- isolation_forest / nasa_ims_1st_test_bearing_3: first_alarm_hour=226.7555555555556, reason=sustained_warning, documented_failure_mode=inner_race_defect
- isolation_forest / nasa_ims_1st_test_bearing_4: first_alarm_hour=671.3286111111112, reason=sustained_warning, documented_failure_mode=roller_element_defect
- isolation_forest / nasa_ims_2nd_test_bearing_1: first_alarm_hour=nan, reason=no_alarm, documented_failure_mode=outer_race_failure
- isolation_forest / nasa_ims_2nd_test_bearing_2: first_alarm_hour=nan, reason=no_alarm, documented_failure_mode=nan
- isolation_forest / nasa_ims_2nd_test_bearing_3: first_alarm_hour=nan, reason=no_alarm, documented_failure_mode=nan
- isolation_forest / nasa_ims_2nd_test_bearing_4: first_alarm_hour=nan, reason=no_alarm, documented_failure_mode=nan
- isolation_forest / nasa_ims_3rd_test__4th_test__txt_bearing_1: first_alarm_hour=nan, reason=no_alarm, documented_failure_mode=nan
- isolation_forest / nasa_ims_3rd_test__4th_test__txt_bearing_2: first_alarm_hour=nan, reason=no_alarm, documented_failure_mode=nan
- isolation_forest / nasa_ims_3rd_test__4th_test__txt_bearing_3: first_alarm_hour=nan, reason=no_alarm, documented_failure_mode=nan
- isolation_forest / nasa_ims_3rd_test__4th_test__txt_bearing_4: first_alarm_hour=nan, reason=no_alarm, documented_failure_mode=nan

## Conservative Interpretation
- Snapshot-level labels remain unknown throughout the experiment.
- Documented end-of-run bearing failures are stored separately as metadata and are not used as dense targets.
- Threshold crossings indicate deviation from the early calibrated regime, not a verified failure prediction time.
