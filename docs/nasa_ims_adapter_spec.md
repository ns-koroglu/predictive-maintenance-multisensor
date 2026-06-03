# NASA IMS Adapter Specification
## Revised Version

## Revision Scope

This revision applies three changes only:

1. Snapshot duration is represented conservatively as a nominal documented duration, not as a stronger asserted physical interval.
2. A `reference_region_role` field is added for anomaly-first experiment compatibility.
3. `channel_indices` remains a logical list in the normalized schema, with explicit CSV serialization guidance.

---

## 1. Scope

This specification defines a strict, compact adapter target for the NASA IMS bearing dataset located under:

- [data/external/nasa_ims/extracted](C:/Users/Enes/Documents/Projeler/predictive-maintenance-multisensor/data/external/nasa_ims/extracted)

The adapter must:

- preserve chronological run-to-failure structure
- preserve bearing-level trajectories
- avoid fabricated labels
- avoid large raw merged exports
- fit the current multi-dataset framework

This specification defines the normalized schema and export targets only.  
It does not define parser code.

---

## 2. Core Modeling Assumptions

## 2.1 Dataset Role

NASA IMS is treated as a:

- run-to-failure dataset
- vibration-only dataset
- anomaly-first dataset
- degradation-sequence dataset

It is not treated as:

- a classification-first dataset
- a densely labeled fault-state dataset
- a synchronized multi-sensor dataset

## 2.2 Granularity

One raw file is one timestamped vibration snapshot.

One normalized session is one bearing trajectory within one test run.

One group is one test run.

This implies:

- `group_id` = test run
- `session_id` = test run + bearing id

---

## 3. Canonical Identifiers

## 3.1 Run Key

Each discovered run path must be assigned a stable normalized run key.

### Allowed run keys for the currently observed local layout

| Source path | Normalized `run_key` |
| --- | --- |
| `1st_test/` | `1st_test` |
| `2nd_test/` | `2nd_test` |
| `3rd_test/4th_test/txt/` | `3rd_test__4th_test__txt` |

Strict rule:

- the adapter must preserve the actual local path identity
- it must not silently rename `3rd_test/4th_test/txt` to `3rd_test`

## 3.2 Group ID

Format:

```text
nasa_ims_<run_key>
```

Examples:

- `nasa_ims_1st_test`
- `nasa_ims_2nd_test`
- `nasa_ims_3rd_test__4th_test__txt`

## 3.3 Session ID

Format:

```text
nasa_ims_<run_key>_bearing_<bearing_id>
```

Examples:

- `nasa_ims_1st_test_bearing_1`
- `nasa_ims_1st_test_bearing_4`
- `nasa_ims_2nd_test_bearing_1`
- `nasa_ims_3rd_test__4th_test__txt_bearing_3`

`bearing_id` must be `1`, `2`, `3`, or `4`.

---

## 4. Exact Normalized Internal Schema

The adapter must output one feature row per:

- `session_id`
- `snapshot_timestamp`

Canonical row-level schema:

```yaml
row:
  dataset_name: "nasa_ims"
  dataset_variant: "compact_bearing_progression"
  dataset_display_name: "NASA IMS Bearing"
  group_id: str
  session_id: str
  label: "unknown"
  multiclass_label: "unknown"
  reference_region_role: "unknown" | "calibration" | "evaluation"
  bearing_id: int
  run_key: str
  source_run_path: str
  source_file_name: str
  source_relative_path: str
  snapshot_timestamp: str
  progression_index: int
  elapsed_minutes: float
  elapsed_hours: float
  relative_progress: float
  nominal_snapshot_duration_sec: 1.0
  window_index: int
  window_start: float
  window_end: float
  window_pairing_strategy: "file_timestamp_order"
  has_vibration: true
  has_thermal: false
  has_current: false
  has_ae: false
  has_acoustic: false
  axis_count: int
  channel_indices: list[int]
  documented_failure_mode: str | null
  documented_failed_bearing: bool | null
  documented_failure_metadata_confidence: "documented" | "uncertain_local_layout"
  ...
  vibration feature columns
```

Rules:

- `window_index = progression_index`
- `window_start = elapsed_minutes * 60.0`
- `nominal_snapshot_duration_sec = 1.0` is a documentation-derived nominal duration
- `window_end` may be set using the nominal duration in downstream experiments if needed, but it must be interpreted as a nominal window end rather than a directly measured duration boundary
- `label` must remain `unknown`
- `multiclass_label` must remain `unknown`
- `reference_region_role` is reserved for anomaly-first experiment use:
  - `unknown` in adapter outputs by default
  - `calibration` or `evaluation` may be assigned later by the experiment layer

---

## 5. Run Manifest Schema

File target:

- `data/processed/nasa_ims/manifests/run_manifest.csv`

One row per discovered run path.

Required fields:

| Field | Type | Description |
| --- | --- | --- |
| `dataset_name` | `string` | Fixed: `nasa_ims` |
| `dataset_variant` | `string` | Fixed: `compact_bearing_progression` |
| `run_key` | `string` | Stable normalized run key |
| `group_id` | `string` | Canonical group identifier |
| `source_run_path` | `string` | Relative run path under extracted root |
| `layout_type` | `string` | `flat_run_folder` or `nested_run_folder` |
| `layout_warning` | `string or null` | Warning text for unusual packaging |
| `n_snapshot_files` | `integer` | Number of files in the run |
| `start_timestamp` | `string` | ISO 8601 start timestamp from first file name |
| `end_timestamp` | `string` | ISO 8601 end timestamp from last file name |
| `nominal_interval_minutes` | `number` | Dominant timestamp delta |
| `min_interval_minutes` | `number` | Minimum observed timestamp delta |
| `max_interval_minutes` | `number` | Maximum observed timestamp delta |
| `sampling_rate_hz` | `number` | Fixed: `20000.0` |
| `nominal_snapshot_duration_sec` | `number` | Fixed nominal/documented value: `1.0` |
| `rows_per_snapshot` | `integer` | Expected: `20480` |
| `channel_count` | `integer` | 8 or 4 |
| `bearing_count` | `integer` | Fixed: `4` |
| `documented_failure_notes` | `string or null` | Readme-derived run outcome text |
| `documented_failure_metadata_confidence` | `string` | `documented` or `uncertain_local_layout` |

---

## 6. Bearing Session Manifest Schema

File target:

- `data/processed/nasa_ims/manifests/bearing_session_manifest.csv`

One row per bearing trajectory.

Required fields:

| Field | Type | Description |
| --- | --- | --- |
| `dataset_name` | `string` | Fixed: `nasa_ims` |
| `dataset_variant` | `string` | Fixed: `compact_bearing_progression` |
| `run_key` | `string` | Run identity |
| `group_id` | `string` | Test-run identifier |
| `session_id` | `string` | Bearing trajectory identifier |
| `bearing_id` | `integer` | 1 to 4 |
| `source_run_path` | `string` | Relative path under extracted root |
| `channel_indices` | `string` | Serialized representation of the logical list of source channel ids, for example `"[1, 2]"` or `"[3]"` |
| `axis_count` | `integer` | 2 for `1st_test`, 1 for later runs |
| `n_snapshots` | `integer` | Number of feature rows for this bearing |
| `start_timestamp` | `string` | First snapshot timestamp |
| `end_timestamp` | `string` | Last snapshot timestamp |
| `sampling_rate_hz` | `number` | Fixed: `20000.0` |
| `nominal_snapshot_duration_sec` | `number` | Fixed nominal/documented value: `1.0` |
| `label` | `string` | Fixed: `unknown` |
| `multiclass_label` | `string` | Fixed: `unknown` |
| `reference_region_role` | `string` | Fixed: `unknown` in adapter outputs |
| `documented_failure_mode` | `string or null` | Run-readme failure mode if this bearing is documented failed |
| `documented_failed_bearing` | `boolean or null` | True, false, or null when uncertain |
| `documented_failure_metadata_confidence` | `string` | `documented` or `uncertain_local_layout` |
| `layout_warning` | `string or null` | Repeated from run-level if applicable |

Logical rule:

- `channel_indices` is logically a list of integers in the normalized internal schema
- when stored in CSV manifests, it may be serialized as a string representation of that list

---

## 7. Channel-to-Bearing Mapping Rules

## 7.1 `1st_test`

Observed channel count: `8`

Mapping:

| Bearing | Source channels | Axis count |
| --- | --- | ---: |
| 1 | `[1, 2]` | 2 |
| 2 | `[3, 4]` | 2 |
| 3 | `[5, 6]` | 2 |
| 4 | `[7, 8]` | 2 |

Export rule:

- preserve two raw vibration axes per bearing
- do not rename them to x/y unless parser confirmation is later added
- use safe names:
  - `vibration_axis_1`
  - `vibration_axis_2`

## 7.2 `2nd_test`

Observed channel count: `4`

Mapping:

| Bearing | Source channels | Axis count |
| --- | --- | ---: |
| 1 | `[1]` | 1 |
| 2 | `[2]` | 1 |
| 3 | `[3]` | 1 |
| 4 | `[4]` | 1 |

Export rule:

- single-axis vibration only
- use:
  - `vibration_axis_1`

## 7.3 Nested `3rd_test/4th_test/txt`

Observed local channel count: `4`

Mapping for the local extracted path:

| Bearing | Source channels | Axis count |
| --- | --- | ---: |
| 1 | `[1]` | 1 |
| 2 | `[2]` | 1 |
| 3 | `[3]` | 1 |
| 4 | `[4]` | 1 |

Strict rule:

- this mapping is based on observed local file structure
- the run identity must remain `3rd_test__4th_test__txt`
- do not assert this is canonical “official set 3” in normalized metadata

---

## 8. Progression Metadata Fields

Chronology must be built only from parsed filename timestamps.

Required per-row fields:

| Field | Type | Rule |
| --- | --- | --- |
| `snapshot_timestamp` | `string` | ISO 8601 from filename |
| `progression_index` | `integer` | Zero-based chronological order within session |
| `elapsed_minutes` | `float` | Minutes since first file in the same session |
| `elapsed_hours` | `float` | `elapsed_minutes / 60.0` |
| `relative_progress` | `float` | `progression_index / (n_snapshots - 1)` when `n_snapshots > 1`, else `0.0` |
| `nominal_snapshot_duration_sec` | `float` | Fixed nominal/documented value: `1.0` |
| `reference_region_role` | `string` | `unknown` at adapter stage; may become `calibration` or `evaluation` at experiment stage |

Additional optional but recommended fields:

| Field | Type | Description |
| --- | --- | --- |
| `delta_minutes_from_previous` | `float` | Gap since prior snapshot |
| `is_nominal_interval` | `boolean` | Whether delta matches dominant run cadence |
| `progression_gap_flag` | `boolean` | Whether delta is unusually large |

Interpretation rule:

- larger gaps indicate acquisition interruptions or next-day resumptions
- they must be preserved as timing metadata, not imputed away

---

## 9. Label Handling Rules

## 9.1 `label`

For all rows and all bearing sessions:

- `label = "unknown"`

Reason:

- there are no dense verified health labels at snapshot level

## 9.2 `multiclass_label`

For all rows and all bearing sessions:

- `multiclass_label = "unknown"`

Reason:

- no dense fault-state timeline is provided

## 9.3 `reference_region_role`

For all adapter outputs by default:

- `reference_region_role = "unknown"`

Experiment-layer usage:

- `reference_region_role = "calibration"` for early healthy-reference rows used to fit anomaly models
- `reference_region_role = "evaluation"` for later rows scored against that reference

Strict rule:

- this field supports anomaly-first experiments
- it must not be confused with a ground-truth health label

## 9.4 Documented Failure Metadata

Readme-derived end-of-run failure metadata must be stored separately from `label`.

Required session-level metadata fields:

| Field | Type | Description |
| --- | --- | --- |
| `documented_failure_mode` | `string or null` | Failure mode for the documented failed bearing |
| `documented_failed_bearing` | `boolean or null` | Whether this bearing is the documented failed bearing |
| `documented_failure_metadata_source` | `string` | `ims_readme_pdf` |
| `documented_failure_metadata_confidence` | `string` | `documented` or `uncertain_local_layout` |
| `documented_failure_notes` | `string or null` | Human-readable note |

## 9.5 Known Run-Level Failure Metadata

### `1st_test`

From local readme:

- bearing 3: inner race defect
- bearing 4: roller element defect

Session metadata assignment:

- `bearing_3`:
  - `documented_failure_mode = "inner_race_defect"`
  - `documented_failed_bearing = true`
  - `documented_failure_metadata_confidence = "documented"`
- `bearing_4`:
  - `documented_failure_mode = "roller_element_defect"`
  - `documented_failed_bearing = true`
  - `documented_failure_metadata_confidence = "documented"`
- `bearing_1`, `bearing_2`:
  - `documented_failure_mode = null`
  - `documented_failed_bearing = false`
  - `documented_failure_metadata_confidence = "documented"`

### `2nd_test`

From local readme:

- bearing 1: outer race failure

Session metadata assignment:

- `bearing_1`:
  - `documented_failure_mode = "outer_race_failure"`
  - `documented_failed_bearing = true`
  - `documented_failure_metadata_confidence = "documented"`
- others:
  - `documented_failure_mode = null`
  - `documented_failed_bearing = false`
  - `documented_failure_metadata_confidence = "documented"`

### Nested `3rd_test/4th_test/txt`

Safe rule:

- do not automatically copy official `Set No. 3` failure metadata onto this local nested path
- set:
  - `documented_failure_mode = null`
  - `documented_failed_bearing = null`
  - `documented_failure_metadata_confidence = "uncertain_local_layout"`

Reason:

- the local extracted run length and path identity do not match the local readme’s documented `Set No. 3`

---

## 10. Safe Handling Of The Nested Packaging Anomaly

The local path:

- `3rd_test/4th_test/txt`

must be represented explicitly as a packaging anomaly.

Required run-level fields:

| Field | Value |
| --- | --- |
| `layout_type` | `nested_run_folder` |
| `layout_warning` | `Local extracted layout contains a nested run path under 3rd_test/4th_test/txt; this run is preserved by observed path identity rather than assumed official set numbering.` |

Strict rules:

- do not collapse it to `3rd_test`
- do not silently drop `4th_test/txt`
- do not assume the official readme’s `Set No. 3` labels apply
- preserve the exact `source_run_path`

This is necessary for thesis-safe provenance.

---

## 11. Compact Feature Dataset Rules

The adapter must produce a compact feature dataset only.

It must not produce:

- huge raw merged snapshot dumps
- dense per-file raw exports for all channels unless explicitly requested later

One feature row corresponds to:

- one timestamped snapshot
- one bearing trajectory

Expected feature families:

- time-domain vibration features
- frequency-domain vibration features
- optional simple trend-aware vibration summary features

No thermal, current, AE, or acoustic fields are required.

Modality flags must be explicit:

- `has_vibration = true`
- `has_thermal = false`
- `has_current = false`
- `has_ae = false`
- `has_acoustic = false`

---

## 12. Compact Processed Output Layout

## 12.1 `data/processed/nasa_ims/`

Required structure:

```text
data/processed/nasa_ims/
  manifests/
    run_manifest.csv
    bearing_session_manifest.csv
    layout_warnings.csv
    documented_failure_map.json
  datasets/
    nasa_ims_bearing_feature_dataset.csv
  adapter_summary.json
  adapter_summary.md
```

### `manifests/run_manifest.csv`
As defined in Section 5.

### `manifests/bearing_session_manifest.csv`
As defined in Section 6.

### `manifests/layout_warnings.csv`
One row per structural anomaly.

Required fields:

- `run_key`
- `group_id`
- `source_run_path`
- `warning_type`
- `warning_detail`

### `manifests/documented_failure_map.json`

Required structure:

```json
{
  "source": "ims_readme_pdf",
  "runs": {
    "1st_test": {
      "bearing_3": "inner_race_defect",
      "bearing_4": "roller_element_defect"
    },
    "2nd_test": {
      "bearing_1": "outer_race_failure"
    },
    "3rd_test__4th_test__txt": {
      "status": "uncertain_local_layout"
    }
  }
}
```

### `datasets/nasa_ims_bearing_feature_dataset.csv`

One row per bearing snapshot.

Required base columns:

- `dataset_name`
- `dataset_variant`
- `dataset_display_name`
- `group_id`
- `session_id`
- `label`
- `multiclass_label`
- `reference_region_role`
- `bearing_id`
- `run_key`
- `source_run_path`
- `source_file_name`
- `source_relative_path`
- `snapshot_timestamp`
- `progression_index`
- `elapsed_minutes`
- `elapsed_hours`
- `relative_progress`
- `nominal_snapshot_duration_sec`
- `window_index`
- `window_start`
- `window_end`
- `window_pairing_strategy`
- `has_vibration`
- `has_thermal`
- `has_current`
- `has_ae`
- `has_acoustic`
- `axis_count`
- `channel_indices`
- `documented_failure_mode`
- `documented_failed_bearing`
- `documented_failure_metadata_confidence`

Then vibration feature columns follow.

Serialization rule:

- in CSV, `channel_indices` may be stored as a string representation of the logical list, for example `"[1, 2]"` or `"[3]"`

---

## 13. Experiment Output Layout

## 13.1 `results/nasa_ims_experiment/`

First experiment type should be anomaly-first only.

Required structure:

```text
results/nasa_ims_experiment/
  config_snapshot.yaml
  experiment_summary.json
  experiment_summary.md
  datasets/
    nasa_ims_bearing_feature_dataset.csv
    healthy_reference_region.csv
  predictions/
    anomaly_trend.csv
    threshold_crossing_summary.csv
  metrics/
    anomaly_model_calibration.csv
    anomaly_model_calibration.json
  plots/
    anomaly_trend.png
  artifacts/
    isolation_forest_artifact.joblib
    one_class_svm_artifact.joblib
  models/
    isolation_forest.joblib
    one_class_svm.joblib
```

This mirrors the current run-to-failure experiment style already used in the framework.

`reference_region_role` usage in experiments:

- `calibration` for rows used to fit the anomaly baseline
- `evaluation` for later chronological scoring rows

---

## 14. Compatibility With The Current Framework

This specification is compatible with the current framework because it preserves:

- `dataset_name`
- `dataset_variant`
- `session_id`
- `group_id`
- row-wise feature dataset exports
- explicit modality flags
- progression metadata
- anomaly-first experiment semantics

It also fits the current run-to-failure pattern:

- one compact feature dataset
- no raw full-sample dump
- early-reference calibration
- chronological anomaly trend outputs

---

## 15. Summary Of Required Adapter Behavior

The future NASA IMS adapter must:

- discover run folders by actual file-bearing leaf paths
- preserve the nested `3rd_test/4th_test/txt` packaging anomaly explicitly
- parse chronology only from filename timestamps
- create one `group_id` per test run
- create one `session_id` per bearing trajectory
- map channels to bearings according to observed run structure
- keep `label` and `multiclass_label` as `unknown`
- add `reference_region_role` for anomaly-first experiment compatibility
- store documented end-of-run failure metadata separately
- mark uncertain failure metadata explicitly for the nested local path
- treat snapshot duration as a nominal documented duration, not a stronger measured fact
- preserve `channel_indices` logically as a list of integers
- export a compact bearing-level feature dataset only
- support anomaly-first experiments, not classification-first claims
