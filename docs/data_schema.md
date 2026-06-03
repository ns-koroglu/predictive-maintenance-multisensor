# Expected Data Schema

This repository expects one folder per measurement session.

## Folder Structure

```text
session_xxx/
  ae.csv
  vibration.csv
  thermal.csv
  metadata.json
```

## `ae.csv`

Required columns:

- `timestamp`
- at least one AE signal column such as `ae`

Notes:

- timestamps should be numeric and monotonically increasing after sorting
- one or multiple AE channels are allowed

## `vibration.csv`

Required columns:

- `timestamp`
- at least one vibration signal column such as `ax`, `ay`, `az`

Notes:

- timestamps should be numeric
- single-axis or multi-axis vibration is supported

## `thermal.csv`

Required columns:

- `timestamp`
- `t_mean`
- `t_max`
- `hotspot_area`

Notes:

- this baseline expects thermal summary values, not raw thermal images
- additional numeric columns are allowed

## `metadata.json`

Recommended fields:

- `session_id`
- `label`
- `rpm`
- `load_level`
- `ambient_temp_c`
- `lubrication_state`
- `notes`

Optional label examples:

- `healthy`
- `developing_fault`
- `faulty`

## Split Validity

For thesis-grade evaluation, the default split strategy is session-aware.

This means:

- all windows from one session stay together
- train and test sessions are disjoint
- window-level splitting is only allowed when explicitly configured
