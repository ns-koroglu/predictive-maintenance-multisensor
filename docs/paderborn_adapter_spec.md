# Paderborn Adapter Specification

## 1. Scope

Bu doküman, Paderborn bearing veri seti için ilk kompakt adapter hedef şemasını tanımlar.

Geçerli veri kökü:
- `data/external/paderborn/extracted`

Bu spesifikasyonun amacı:
- veri setini mevcut multi-dataset framework ile uyumlu hale getirmek
- semantik belirsizlikleri gizlememek
- büyük ham export üretmeden kompakt feature dataset çıkarmak
- ilk benchmark'ı açıkça tanımlamak

Bu doküman parser kodu içermez.

## 2. Core Positioning

Paderborn veri seti bu projede şu şekilde konumlanır:

- snapshot tabanlı bearing condition veri seti
- run-to-failure veri seti değildir
- ilk benchmark için classification-first uygundur
- ilk benchmark ikili sınıflandırmadır
- ilk benchmark vibration-only olmalıdır
- ek modaliteler korunur ama ilk baseline'a zorunlu dahil edilmez

İlk benchmark:
- `binary healthy vs faulty`
- `vibration-only classification baseline`
- `group-aware split` bearing code bazında

## 3. Canonical Identity Rules

### 3.1 Bearing Code

Bearing code, kök klasör adından gelir.

Örnekler:
- `K001`
- `KA01`
- `KB23`
- `KI01`

### 3.2 Session Identity

Strict rule:
- `session_id = one measurement file`

Önerilen format:

```text
paderborn_<bearing_code>_<condition_code>_<replicate_index_2d>
```

Örnekler:
- `paderborn_K001_N09_M07_F10_01`
- `paderborn_KA01_N15_M01_F10_12`
- `paderborn_KB23_N15_M07_F04_20`

### 3.3 Group Identity

Strict rule:
- `group_id = one bearing code`
- `split_group = group_id`

Önerilen format:

```text
paderborn_<bearing_code>
```

Örnekler:
- `paderborn_K001`
- `paderborn_KA01`
- `paderborn_KB23`

Gerekçe:
- aynı bearing code'a ait farklı operating condition ve tekrarlar train/test arasında karışmamalıdır.

## 4. First Benchmark Definition

İlk benchmark açıkça aşağıdaki görevdir:

- görev türü: `binary classification`
- hedef: `healthy vs faulty`
- giriş modalitesi: `vibration only`
- split mantığı: `group-aware`
- split birimi: `bearing code`

İlk benchmark'ta:
- `current`
- `temperature`
- `force`
- `speed`
- `torque`

korunur ama ilk sınıflandırma baseline'ına dahil edilmez.

## 5. Local Structure Representation Rules

### 5.1 Folder Layout

Beklenen yerel yapı:

```text
data/external/paderborn/extracted/
  K001/
  K002/
  K003/
  ...
  KI21/
```

Çoğu klasörde:
- `80` `.mat`
- `2` `.pdf`

bulunur.

### 5.2 K002 Exception

`K002` için nested yapı açıkça korunmalıdır.

Güvenli temsil:
- `source_folder_layout = nested_single_subfolder`
- `layout_warning` alanında bu durum kaydedilmelidir

Strict rule:
- `K002` düz klasörmüş gibi sessizce normalize edilmemeli
- fakat adapter düzeyinde dosya keşfi bunu güvenle desteklemelidir

## 6. Normalized Internal Row Schema

Bir `.mat` dosyası bir kompakt feature row üretir.

Canonical row schema:

```yaml
row:
  dataset_name: "paderborn"
  dataset_variant: "compact_multirate_snapshot"
  dataset_display_name: "Paderborn Bearing"
  session_id: str
  group_id: str
  split_group: str
  bearing_code: str
  condition_code: str
  operating_condition_n: str
  operating_condition_m: str
  operating_condition_f: str
  replicate_index: int
  source_file_name: str
  source_relative_path: str
  source_folder_layout: "flat_bearing_folder" | "nested_single_subfolder"
  layout_warning: str | null
  label: "healthy" | "faulty"
  multiclass_label: "unknown"
  fault_component_normalized: "none" | "outer_ring" | "inner_ring" | "compound_mixed" | "ambiguous"
  fault_origin_normalized: "none" | "artificial" | "fatigue" | "plastic_deformation" | "mixed" | "unknown"
  fault_component_raw: str | null
  fault_origin_raw: str | null
  documented_fault_notes: str | null
  has_vibration: bool
  has_current: bool
  has_thermal: bool
  has_force: bool
  has_speed: bool
  has_torque: bool
  nominal_record_duration_sec: 4.0
  sampling_rate_vibration_hz: 64000.0 | null
  sampling_rate_current_hz: 64000.0 | null
  sampling_rate_mechanical_hz: 4000.0 | null
  sampling_rate_temperature_hz: 1.0 | null
  ...
  feature columns
```

Strict rules:
- `multiclass_label` ilk adapter sürümünde `unknown` kalmalıdır
- gerçek fault semantics, `fault_component_normalized` ve `fault_origin_normalized` alanlarında tutulmalıdır
- bir feature row, kronolojik pencere değil, kısa snapshot ölçümüdür

## 7. Bearing Manifest Schema

Dosya hedefi:
- `data/processed/paderborn/manifests/bearing_manifest.csv`

Bir satır = bir bearing code

Gerekli alanlar:

| Field | Type | Description |
| --- | --- | --- |
| `dataset_name` | string | Fixed: `paderborn` |
| `dataset_variant` | string | Fixed: `compact_multirate_snapshot` |
| `bearing_code` | string | Raw bearing code |
| `group_id` | string | Canonical group id |
| `split_group` | string | Must equal `group_id` |
| `bearing_family_prefix` | string | `K`, `KA`, `KB`, `KI` |
| `label` | string | `healthy` or `faulty` |
| `multiclass_label` | string | Fixed: `unknown` in first version |
| `fault_component_normalized` | string | Conservative normalized component class |
| `fault_origin_normalized` | string | Conservative normalized origin class |
| `fault_component_raw` | string or null | Raw component text from PDF |
| `fault_origin_raw` | string or null | Raw mode text from PDF |
| `n_recordings` | integer | Expected: `80` |
| `n_conditions` | integer | Expected: `4` |
| `n_profile_pdfs` | integer | Expected: `1` |
| `n_measuring_log_pdfs` | integer | Expected: `1` |
| `source_folder_layout` | string | Flat or nested |
| `layout_warning` | string or null | Structural warning, especially for `K002` |
| `normalization_warning_count` | integer | Number of semantic warnings |
| `documented_fault_notes` | string or null | Short bearing profile summary |

## 8. Recording Manifest Schema

Dosya hedefi:
- `data/processed/paderborn/manifests/recording_manifest.csv`

Bir satır = bir `.mat` ölçüm dosyası

Gerekli alanlar:

| Field | Type | Description |
| --- | --- | --- |
| `dataset_name` | string | Fixed: `paderborn` |
| `dataset_variant` | string | Fixed: `compact_multirate_snapshot` |
| `session_id` | string | One measurement file |
| `group_id` | string | One bearing code |
| `split_group` | string | Must equal `group_id` |
| `bearing_code` | string | Bearing code |
| `condition_code` | string | Example: `N09_M07_F10` |
| `operating_condition_n` | string | Example: `N09` |
| `operating_condition_m` | string | Example: `M07` |
| `operating_condition_f` | string | Example: `F10` |
| `replicate_index` | integer | 1 to 20 |
| `source_file_name` | string | Original file name |
| `source_relative_path` | string | Relative path under extracted root |
| `source_folder_layout` | string | Flat or nested |
| `layout_warning` | string or null | Explicit structural warning |
| `label` | string | `healthy` or `faulty` |
| `multiclass_label` | string | Fixed: `unknown` |
| `fault_component_normalized` | string | Conservative normalized component class |
| `fault_origin_normalized` | string | Conservative normalized origin class |
| `fault_component_raw` | string or null | Raw component field from profile PDF |
| `fault_origin_raw` | string or null | Raw mode field from profile PDF |
| `has_vibration` | boolean | Expected true |
| `has_current` | boolean | Expected true |
| `has_thermal` | boolean | Expected true |
| `has_force` | boolean | Expected true |
| `has_speed` | boolean | Expected true |
| `has_torque` | boolean | Expected true |
| `nominal_record_duration_sec` | number | Fixed nominal value `4.0` |
| `sampling_rate_vibration_hz` | number or null | Expected `64000.0` |
| `sampling_rate_current_hz` | number or null | Expected `64000.0` |
| `sampling_rate_mechanical_hz` | number or null | Expected `4000.0` |
| `sampling_rate_temperature_hz` | number or null | Expected `1.0` |
| `record_status` | string | `ok` or parse/validation status |

## 9. Label Normalization Rules

### 9.1 Binary Label

İlk benchmark için ikili etiket açıkça tanımlanır:

- `Kxxx` -> `healthy`
- `KAxx`, `KBxx`, `KIxx` -> `faulty`

### 9.2 Multiclass Label

İlk adapter sürümünde:

- `multiclass_label = "unknown"`

### 9.3 Fault Component Normalization

Alan:
- `fault_component_normalized`

İzinli değerler:
- `none`
- `outer_ring`
- `inner_ring`
- `compound_mixed`
- `ambiguous`

Kurallar:
- healthy bearings (`Kxxx`) -> `none`
- yalnız `OR` içeriyorsa -> `outer_ring`
- yalnız `IR` içeriyorsa -> `inner_ring`
- hem `IR` hem `OR` varsa -> `compound_mixed`
- `AR` gibi açık olmayan yerel durumlar -> `ambiguous`

### 9.4 Fault Origin Normalization

Alan:
- `fault_origin_normalized`

İzinli değerler:
- `none`
- `artificial`
- `fatigue`
- `plastic_deformation`
- `mixed`
- `unknown`

Kurallar:
- healthy bearings (`Kxxx`) -> `none`
- tek mode `artificial` -> `artificial`
- tek mode `fatigue` -> `fatigue`
- tek mode `plastic deformation` -> `plastic_deformation`
- birden fazla mode birlikte geçiyorsa -> `mixed`
- PDF'den güvenli biçimde okunamıyorsa -> `unknown`

### 9.5 Raw Semantics Preservation

Şu iki alan zorunludur:
- `fault_component_raw`
- `fault_origin_raw`

## 10. Operating Condition Normalization Rules

Condition code dosya adından gelir:

```text
Nxx_Mxx_Fxx
```

Gerekli alanlar:
- `condition_code`
- `operating_condition_n`
- `operating_condition_m`
- `operating_condition_f`

Conservative interpretation:
- bu alanlar operating condition token'larıdır
- fiziksel açılımları adapter içinde sabit bilgi gibi yazılmamalıdır

## 11. Modality Availability Rules

Yerel `.mat` incelemesine göre güvenilir kanallar:

- vibration: `vibration_1`
- current: `phase_current_1`, `phase_current_2`
- thermal: `temp_2_bearing_module`
- force: `force`
- speed: `speed`
- torque: `torque`

Bu nedenle adapter şu alanları açıkça üretmelidir:

- `has_vibration = true`
- `has_current = true`
- `has_thermal = true`
- `has_force = true`
- `has_speed = true`
- `has_torque = true`

Strict rule:
- ambient temperature yoksa uydurulmamalı
- AE yoksa uydurulmamalı
- acoustic yoksa uydurulmamalı

## 12. Compact Feature Dataset Rules

Adapter tam ham export üretmemelidir.

Strict rule:
- one `.mat` file -> one compact feature row

İlk kompakt dataset:
- vibration özelliklerini içermeli
- current / thermal / mechanical modalite özelliklerini korumalı
- fakat ilk benchmark için feature-selection veya subset ile vibration-only kullanımına izin vermeli

Önerilen feature blokları:
- `vibration_*`
- `current_*`
- `thermal_*`
- `force_*`
- `speed_*`
- `torque_*`

İlk benchmark için:
- yalnız `vibration_*` sütunları kullanılmalıdır

## 13. Compact Processed Output Layout

Hedef kök:
- `data/processed/paderborn`

Yapı:

```text
data/processed/paderborn/
  manifests/
    bearing_manifest.csv
    recording_manifest.csv
    label_normalization_audit.csv
    fault_metadata_map.json
    layout_warnings.csv
  datasets/
    paderborn_compact_feature_dataset.csv
  adapter_summary.json
  adapter_summary.md
```

## 14. First Benchmark Output Layout

İlk benchmark sonuç kökü:
- `results/paderborn_baseline_experiment`

Önerilen yapı:

```text
results/paderborn_baseline_experiment/
  config_snapshot.yaml
  experiment_summary.json
  experiment_summary.md
  metrics/
    classifier_metrics.json
    classifier_metrics.csv
  plots/
    class_distribution.png
    confusion_matrix.png
    feature_importance.png
  predictions/
    per_group_summary.csv
    per_session_predictions.csv
  artifacts/
    feature_columns.json
    classifier_artifact.joblib
    classifier_preprocessor.joblib
    classifier_label_encoder.joblib
  models/
    random_forest_model.joblib
```

İlk benchmark için açık kurallar:
- problem: `binary healthy vs faulty`
- features: `vibration-only`
- split: `group-aware`
- group unit: `bearing_code`
- classifier: `Random Forest`
- raporlanan ana metrikler:
  - `balanced_accuracy`
  - `macro_precision`
  - `macro_recall`
  - `macro_f1`

## 15. Group-Aware Evaluation Rule

En kritik kural:

- aynı `bearing_code` train ve test tarafında birlikte bulunamaz

Strict implementation rule:
- `split_group = group_id`
- `group_id = paderborn_<bearing_code>`

## 16. Known Risks And Ambiguities

Adapter bunları gizlememelidir:

- `K002` nested klasör sapması vardır
- `KA08` içinde `Component AR` ifadesi vardır ve belirsizdir
- `KI04`, `KI14`, `KB23`, `KB24`, `KB27` gibi örneklerde mixed component yapısı vardır
- prefix ile gerçek fault semantics birebir eş değildir
- `Unit` alanları `.mat` içinde çoğu kanalda boş olabilir
- `N/M/F` condition token'larının fiziksel açılımı yerel gözlemle tahmin edilebilir ama adapter'da kesin bilgi gibi kodlanmamalıdır
- veri seti kronolojik progression veri seti değildir

## 17. Recommended First Benchmark Task

İlk resmi Paderborn benchmark şu olmalıdır:

- `binary healthy vs faulty`
- `vibration-only classification baseline`
- `group-aware split by bearing code`

İkinci aşama:
- vibration-only anomaly analysis

Üçüncü aşama:
- curated multiclass benchmark
  - ancak yalnız PDF-tabanlı normalized semantics üretildikten sonra

## 18. Summary Of Required Adapter Behavior

Gelecekteki Paderborn adapter şunları yapmalıdır:

- `30` bearing-code klasörünü keşfetmek
- `K002` nested yapısını açık warning ile temsil etmek
- bir `.mat` dosyasını bir compact feature row'a çevirmek
- `session_id = one measurement file`
- `group_id = one bearing code`
- `split_group = group_id`
- ilk benchmark için `healthy vs faulty` binary etiketi üretmek
- `fault_component_normalized` ve `fault_origin_normalized` alanlarını ayrı tutmak
- multiclass semantics'i kör biçimde prefix'ten türetmemek
- vibration-only ilk baseline'ı desteklemek
- current, temperature, force, speed, torque modalitelerini açıkça korumak
- büyük ham CSV export üretmemek
- publication-friendly compact manifests ve metrics outputs üretmek
