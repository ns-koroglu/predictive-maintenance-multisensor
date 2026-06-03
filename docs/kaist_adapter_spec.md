# KAIST Adapter Specification

## Scope

This document defines the approved first adapter target for the KAIST rotating machine dataset under:

- `data/external/kaist_rotating_machine/extracted`

The specification preserves these constraints:

- the first integration baseline uses `vibration + thermal` only
- `current` is exported and preserved, but it is not required in the first training baseline
- `acoustic` is not AE and remains a separate optional branch
- sessions are condition-matched and unsynchronized
- labels come from normalized filenames
- no fabricated synchronization
- no fabricated AE
- no derived thermal summary columns inside `thermal.csv`

## Canonical Session Definition

One KAIST session represents one normalized operating condition:

- load condition
- fault family
- fault severity

Canonical logical schema:

```yaml
session:
  session_id: str
  condition_key: str
  dataset_name: "kaist_rotating_machine"
  label: "healthy" | "faulty"
  multiclass_label: "normal" | "bpfi" | "bpfo" | "misalignment" | "unbalance"
  label_source: "normalized_filename"
  load_code: "0Nm" | "2Nm" | "4Nm"
  load_nm: float
  fault_family: "normal" | "bpfi" | "bpfo" | "misalignment" | "unbalance"
  fault_family_raw: str
  severity_code: str
  severity_value: float
  severity_unit: "none" | "mm" | "mg"
  condition_detail_label: str
  available_modalities: list[str]
  missing_modalities: list[str]
  sync_status: "condition_matched_unsynchronized"
  shared_timebase: false
  cross_modality_alignment_allowed: false
  modality_info: dict
  source_files: dict
  normalization_warnings: list[str]
```

Primary branch modalities:

- `vibration`
- `thermal`

Preserved but not required in the first training baseline:

- `current`

Optional separate branch modality:

- `acoustic`

## Naming Rules

### Condition Key

Format:

```text
<load_code_lower>_<fault_family>[ _<severity_code_lower> ]
```

Examples:

- `0nm_normal`
- `2nm_bpfi_03`
- `4nm_bpfo_30`
- `0nm_misalignment_01`
- `4nm_unbalance_3318mg`

### Primary Session ID

```text
kaist_<condition_key>
```

Examples:

- `kaist_0nm_normal`
- `kaist_2nm_bpfi_03`

### Optional Acoustic Session ID

```text
kaist_acoustic_<condition_key>
```

Examples:

- `kaist_acoustic_0nm_normal`
- `kaist_acoustic_0nm_bpfi_03`

This avoids collisions with the primary branch.

## Label Normalization

### Load

| Raw | `load_code` | `load_nm` |
| --- | --- | ---: |
| `0Nm` | `0Nm` | `0.0` |
| `2Nm` | `2Nm` | `2.0` |
| `4Nm` | `4Nm` | `4.0` |

### Fault Family

| Raw | Normalized |
| --- | --- |
| `Normal` | `normal` |
| `BPFI` | `bpfi` |
| `BPFO` | `bpfo` |
| `Misalign` | `misalignment` |
| `Unbalance` | `unbalance` |
| `Unbalalnce` | `unbalance` |

Notes:

- `Unbalalnce` is a known typo and must be normalized, but logged.
- filenames remain the label source of truth.

### Severity

Healthy:

| Raw | `severity_code` | `severity_value` | `severity_unit` |
| --- | --- | ---: | --- |
| `Normal` | `none` | `0.0` | `none` |

BPFI / BPFO:

| Raw | `severity_code` | `severity_value` | `severity_unit` |
| --- | --- | ---: | --- |
| `03` | `03` | `0.3` | `mm` |
| `10` | `10` | `1.0` | `mm` |
| `30` | `30` | `3.0` | `mm` |

Misalignment:

| Raw | `severity_code` | `severity_value` | `severity_unit` |
| --- | --- | ---: | --- |
| `01` | `01` | `0.1` | `mm` |
| `03` | `03` | `0.3` | `mm` |
| `05` | `05` | `0.5` | `mm` |

Unbalance:

| Raw | `severity_code` | `severity_value` | `severity_unit` |
| --- | --- | ---: | --- |
| `0583mg` | `0583mg` | `583.0` | `mg` |
| `1169mg` | `1169mg` | `1169.0` | `mg` |
| `1751mg` | `1751mg` | `1751.0` | `mg` |
| `2239mg` | `2239mg` | `2239.0` | `mg` |
| `3318mg` | `3318mg` | `3318.0` | `mg` |

### Binary And Multiclass Labels

- `label = "healthy"` only for `fault_family == "normal"`
- otherwise `label = "faulty"`
- `multiclass_label = fault_family`

Examples:

- `normal`
- `bpfi`
- `bpfo`
- `misalignment`
- `unbalance`

## `metadata.json`

Required top-level fields:

- `schema_version`: `kaist_adapter_v1`
- `dataset_name`: `kaist_rotating_machine`
- `session_id`
- `condition_key`
- `label`
- `multiclass_label`
- `label_source`: `normalized_filename`
- `load_code`
- `load_nm`
- `fault_family`
- `fault_family_raw`
- `severity_code`
- `severity_value`
- `severity_unit`
- `condition_detail_label`
- `available_modalities`
- `missing_modalities`
- `sync_status`: `condition_matched_unsynchronized`
- `shared_timebase`: `false`
- `cross_modality_alignment_allowed`: `false`
- `source_files`
- `modality_info`
- `normalization_warnings`

`source_files` keys:

- `vibration`
- `current_temp`
- `acoustic`

`modality_info.<modality>` fields:

- `present`
- `export_file`
- `source_format`
- `channel_names`
- `channel_count`
- `sample_rate_hz`
- `duration_s`
- `timestamp_origin`: `local_relative_seconds`
- `absolute_start_time`
- `units`

Missing modalities use:

- `present = false`
- `export_file = null`
- `source_format = null`
- `channel_names = []`
- `channel_count = 0`
- `sample_rate_hz = null`
- `duration_s = null`
- `absolute_start_time = null`
- `units = {}`

## CSV Export Schemas

General rules:

- UTF-8 CSV
- `timestamp` is local relative seconds from that modality's own start time
- `sample_index` is zero-based integer
- rows are sorted by ascending `timestamp`
- no absolute timestamps are written into CSV

### `vibration.csv`

Required columns:

- `timestamp`
- `sample_index`
- `vibration_point_1_g`
- `vibration_point_2_g`
- `vibration_point_3_g`
- `vibration_point_4_g`

### `thermal.csv`

Required columns:

- `timestamp`
- `sample_index`
- `temp_channel_1_c`
- `temp_channel_2_c`

Strict rules:

- do not export `t_mean`
- do not export `t_max`
- derived thermal summary values must be computed later during feature extraction
- do not fabricate `hotspot_area`

### `current.csv`

Required columns:

- `timestamp`
- `sample_index`
- `current_channel_1_a`
- `current_channel_2_a`
- `current_channel_3_a`

Notes:

- exported and preserved
- not required in the first training baseline

### Optional `acoustic.csv`

Required columns:

- `timestamp`
- `sample_index`
- `acoustic_pa`

Strict rules:

- never rename `acoustic` to `ae`
- never export `ae.csv` from KAIST acoustic data

## Unsynchronized Modalities

Because the dataset is condition-matched but not safely time-synchronous:

- each modality keeps its own local `timestamp`
- no merged timestamp grid is created
- no cross-modality resampling is performed in the adapter
- sessions may only be joined by condition identity
- `sync_status` stays `condition_matched_unsynchronized`
- `shared_timebase = false`
- `cross_modality_alignment_allowed = false`

## Missing Modalities

Missing modalities must be represented by absence:

- do not create empty placeholder CSV files
- do not fabricate values
- do not fabricate AE

Representation:

- omit the file
- list the modality in `missing_modalities`
- keep `modality_info.<modality>.present = false`
- keep `source_files.<modality> = null`

## Export Targets

### Interim

```text
data/interim/kaist_rotating_machine/
  inventory.csv
  normalization_audit.csv
  vibration/
  thermal/
  current/
  acoustic/
  source_metadata/
```

### Processed

```text
data/processed/kaist_rotating_machine/
  manifests/
    sessions_manifest.csv
    modality_availability.csv
    label_map.json
  primary_sessions/
    <session_id>/
      metadata.json
      vibration.csv
      thermal.csv
      current.csv
  optional_acoustic_sessions/
    <session_id>/
      metadata.json
      acoustic.csv
```

## First Integration Baseline

The first KAIST training baseline must use:

- `vibration`
- `thermal`

`current` must still be exported and preserved for later studies.

`acoustic` remains a separate optional branch because:

- it is not AE
- it must not be forced into the first baseline

## Required Adapter Behavior

The adapter must:

- normalize labels from filenames only
- create condition-based session IDs
- export `vibration.csv` and `thermal.csv` for the first baseline
- export and preserve `current.csv`
- export optional `acoustic.csv` separately
- preserve `multiclass_label` and binary `label`
- preserve unsynchronized status explicitly
- preserve missing modalities explicitly
- never fabricate AE
- never fabricate synchronization
- never precompute thermal summary columns in `thermal.csv`
