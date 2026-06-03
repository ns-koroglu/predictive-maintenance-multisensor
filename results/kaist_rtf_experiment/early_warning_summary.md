# Early Warning Summary: kaist_rtf_experiment

## Calibration Region
- The anomaly models were trained only on the early reference region of the run.
- This region is treated as the healthy operating baseline for score normalization and threshold calibration.
- kaist_rtf_vibration_bearing_runtofailure: calibration files=24, hours=0.0 to 23.0, mode=`first_24_hours`

## Early Warning Results
### Isolation Forest
- Calibrated threshold: 0.711477
- First threshold crossing: index=nan, hour=nan
- First sustained warning: index=nan, hour=nan
- First reported alarm: index=nan, hour=nan
- Alarm reason: `no_alarm`
- Final rolling anomaly-flag ratio: 0.000
### One Class Svm
- Calibrated threshold: 0.022606
- First threshold crossing: index=24.0, hour=24.0
- First sustained warning: index=26.0, hour=26.0
- First reported alarm: index=26.0, hour=26.0
- Alarm reason: `sustained_warning`
- Final rolling anomaly-flag ratio: 1.000

## Conservative Reporting Note
- These alarm points mark when the run first deviated from the calibrated healthy-reference regime.
- They should not be described as verified failure-prediction times unless external failure-onset evidence is added later.
