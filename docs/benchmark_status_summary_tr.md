# Çoklu Veri Seti Benchmark Durum Özeti

## 1. Çalışmanın Teknik Amacı

Bu çalışmanın teknik amacı, çoklu sensörlü kestirimci bakım için erken arıza uyarısı ve anomali tespiti odaklı, tezde savunulabilir bir deney altyapısı oluşturmaktır. Proje, tek bir yüksek accuracy sonucunu hedeflemekten ziyade, küçük veri, sınıf dengesizliği ve oturum bazlı veri kaçağı riskleri altında daha güvenilir değerlendirme yapmayı hedefler.

Ana yaklaşım feature-level fusion olarak kurgulanmıştır: farklı sensörlerden çıkarılan öznitelikler aynı deney şemasında birleştirilir ve klasik makine öğrenmesi modelleriyle değerlendirilir. Supervised classification yalnızca label yapısı yeterince temiz olduğunda benchmark olarak kullanılır. Dense label bulunmayan run-to-failure veya progression veri setlerinde ise anomaly-first değerlendirme tercih edilir.

Tez açısından kritik ilke, raw accuracy değerini tek başına başarı göstergesi olarak kullanmamaktır. Özellikle ağır sınıf dengesizliği olan veri setlerinde balanced accuracy, macro F1, confusion matrix, threshold davranışı ve session/group-safe split sonuçları birlikte yorumlanmalıdır.

## 2. Kullanılan Veri Setleri

| Veri seti | Sensörler | Görev tipi | Label durumu | Split stratejisi | Tezdeki rolü |
|---|---|---|---|---|---|
| Smoke-test | AE, vibration, thermal | Demo / pipeline doğrulama | Sentetik/örnek `healthy` ve `developing_fault` etiketleri | Session split | Gerçek benchmark değil; uçtan uca pipeline, fusion, açıklanabilirlik ve raporlama doğrulaması |
| KAIST rotating machine | Vibration + thermal ilk baseline; current korunuyor; acoustic opsiyonel ve AE değildir | Binary classification + anomaly-first analiz | Filename normalization ile `healthy/faulty`; çok dengesiz | Session split | Ağır sınıf dengesizliğinde supervised classifier başarısızlığını ve anomaly-first gerekçesini göstermek |
| KAIST run-to-failure | Vibration x/y, bearing temperature, ambient temperature | Chronological anomaly/degradation trend | Dense supervised label yok; erken bölüm healthy-reference olarak kullanılıyor | Tek trajectory içinde kronolojik calibration/evaluation ayrımı | Erken referans bölgesinden sapma ve threshold crossing davranışını göstermek |
| NASA IMS | Vibration | Bearing-level anomaly/degradation analysis | Snapshot-level dense label yok; documented end-of-run failure metadata ayrı tutuluyor | Bearing session + test-run group semantiği; kronolojik calibration/evaluation | Dense label uydurmadan anomaly-first progression analizi göstermek |
| Paderborn | İlk benchmark vibration-only; current/thermal/process kanalları korunuyor | Binary supervised classification | `healthy/faulty` label güçlü; multiclass fault family bu sprintte kullanılmıyor | Group split; `split_group = bearing_code` | İlk temiz supervised public benchmark |

Kullanılan temel kanıt dosyaları: `results/raw_smoke_real_data_readiness/demo_summary.md`, `results/kaist_baseline_experiment/experiment_summary.md`, `results/kaist_rtf_experiment/experiment_summary.md`, `results/nasa_ims_experiment/experiment_summary.md`, `results/paderborn_baseline_experiment/experiment_summary.md`.

## 3. Genel Pipeline Durumu

| Aşama | Durum | Kanıt dosya/klasör | Tez açısından anlamı |
|---|---|---|---|
| Data loading | Var | `data/processed/kaist_run_to_failure/adapter_summary.json`, `data/processed/nasa_ims/adapter_summary.json`, `data/processed/paderborn/adapter_summary.json` | Public datasetler compact processed formata dönüştürülmüş |
| Windowing | Var | `results/raw_smoke_real_data_readiness/demo_summary.md`, `results/kaist_baseline_experiment/datasets/train_windows.csv`, `results/kaist_baseline_experiment/datasets/test_windows.csv` | Zaman serisi verilerinden pencere bazlı feature üretimi mevcut |
| Feature extraction | Var | `results/kaist_feature_build/datasets/kaist_vibration_thermal_features.csv`, `data/processed/paderborn/datasets/paderborn_compact_feature_dataset.csv` | Klasik ML için açık, tezde anlatılabilir öznitelikler kullanılıyor |
| Feature-level fusion | Var | `results/raw_smoke_real_data_readiness/ablation_comparison.csv`, `results/raw_smoke_real_data_readiness/demo_summary.md` | Çoklu sensör birleşimi demo seviyesinde doğrulanmış |
| Session/group-safe split | Var | `results/kaist_baseline_experiment/metrics/split_summary.json`, `results/paderborn_baseline_experiment/datasets/group_split_manifest.csv` | Aynı session veya bearing code'un train/test arasında sızması engelleniyor |
| Random Forest classification | Var | `results/kaist_baseline_experiment/metrics/classifier_metrics.json`, `results/paderborn_baseline_experiment/metrics/classifier_metrics.json` | Supervised baseline klasik ve açıklanabilir düzeyde tutulmuş |
| Isolation Forest / One-Class SVM anomaly scoring | Var | `results/kaist_baseline_experiment/tables/anomaly_metrics_table.md`, `results/kaist_rtf_experiment/anomaly_model_summary.csv`, `results/nasa_ims_experiment/anomaly_model_summary.csv` | Anomaly-first tez hattı destekleniyor |
| Threshold sweep | Var | `results/kaist_baseline_experiment/threshold_sweep.csv`, `results/kaist_baseline_experiment/tables/threshold_sweep_summary_table.md` | Threshold seçiminin performansı ciddi etkilediği gösteriliyor |
| Feature importance / explainability | Var | `results/raw_smoke_real_data_readiness/top_features.csv`, `results/raw_smoke_real_data_readiness/sensor_group_importance.csv`, `results/paderborn_baseline_experiment/plots/feature_importance.png` | SHAP gibi ağır kütüphane olmadan RF feature importance ile açıklama sağlanıyor |
| Report generation | Var | `docs/faculty_presentation_package_tr/`, `src/build_faculty_presentation_package_tr.py`, `src/export_faculty_package_docx_tr.py` | Hoca/tez sunumu için rapor çıktıları üretilmiş |

## 4. Paderborn Supervised Benchmark Özeti

Paderborn, mevcut projede ilk temiz supervised public benchmark olarak konumlandırılmalıdır. Kullanılan deney `results/paderborn_baseline_experiment/experiment_summary.md` dosyasında binary `healthy` vs `faulty` olarak tanımlanmıştır. İlk baseline yalnızca vibration özniteliklerini kullanır ve Random Forest classifier ile çalışır.

Deney kurulumu:

- Task type: binary supervised classification.
- Sensor modality: vibration-only.
- Split type: group-aware split; `split_group = bearing_code`.
- Train groups: 22, test groups: 8.
- Train sessions: 1759, test sessions: 640.
- Group overlap: none.
- Processed dataset dağılımı: `data/processed/paderborn/adapter_summary.json` dosyasına göre 2399 feature row; `faulty`: 1919, `healthy`: 480.
- Deney train/test dağılımı: train `faulty`: 1439, `healthy`: 320; test `faulty`: 480, `healthy`: 160.

Ana metrikler `results/paderborn_baseline_experiment/metrics/classifier_metrics.json` dosyasından alınmıştır:

| Metrik | Değer |
|---|---:|
| Accuracy | 0.7031 |
| Balanced accuracy | 0.5917 |
| Macro precision | 0.5967 |
| Macro recall | 0.5917 |
| Macro F1 | 0.5938 |
| ROC-AUC | 0.8358 |
| PR-AUC | 0.9509 |

Confusion matrix, label sırası `faulty`, `healthy` olacak şekilde:

| True \ Predicted | faulty | healthy |
|---|---:|---:|
| faulty | 391 | 89 |
| healthy | 101 | 59 |

Bu sonuç raw accuracy açısından makul görünse de tezde headline olarak accuracy kullanılmamalıdır. Çünkü test setinde faulty sınıfı çoğunluktadır ve healthy sınıfında recall daha zayıftır. Healthy sınıfı için raporlanan F1 değeri 0.3831'dir. Bu nedenle Paderborn sonucu, güçlü bir nihai sınıflandırıcı iddiasından çok, leakage-safe ve vibration-only supervised baseline olarak sunulmalıdır.

Per-group yorum `results/paderborn_baseline_experiment/predictions/per_group_summary.csv` dosyasından yapılabilir. Test grupları sekiz bearing code içerir. `paderborn_K001` ve `paderborn_K006` healthy olmasına rağmen majority prediction `faulty` çıkmıştır; bu false positive riskini gösterir. `paderborn_KI05` faulty olmasına rağmen majority prediction `healthy` çıkmıştır; bu da bearing-code bazlı genellemenin hâlâ zor olduğunu gösterir. Bu nedenle sonuçlar conservative olarak, group-aware split altında orta düzey supervised baseline performansı şeklinde yorumlanmalıdır.

## 5. KAIST Rotating Machine Analizi

KAIST rotating machine deneyinde ilk baseline vibration + thermal öznitelikleriyle kurulmuştur. Current verisi compact metadata içinde korunmuştur, ancak ilk baseline'da kullanılmamıştır. KAIST acoustic branch AE olarak adlandırılmamalıdır; acoustic ayrı bir opsiyonel dal olarak tutulur. `results/kaist_baseline_experiment/experiment_summary.md` dosyasına göre session semantiği condition-matched fakat açıkça unsynchronized olarak belirtilmiştir.

Deneyde session-safe split kullanılmıştır:

- Train windows: 3352.
- Test windows: 1058.
- Train sessions: 34.
- Test sessions: 11.
- Session overlap: none.
- Train label counts before balancing: `faulty`: 2936, `healthy`: 416.
- Test label counts: `faulty`: 940, `healthy`: 118.
- Classifier train-time balancing: `downsample_majority`, kullanılan train dağılımı `healthy`: 416, `faulty`: 416.

Classifier sonucu supervised classification açısından zayıftır:

| Metrik | Değer |
|---|---:|
| Accuracy | 0.8885 |
| Balanced accuracy | 0.5000 |
| Macro precision | 0.4442 |
| Macro recall | 0.5000 |
| Macro F1 | 0.4705 |
| ROC-AUC | 0.3956 |
| PR-AUC | 0.8655 |

Confusion matrix `results/kaist_baseline_experiment/metrics/classifier_confusion_matrix.csv` dosyasına göre:

| True \ Predicted | faulty | healthy |
|---|---:|---:|
| faulty | 940 | 0 |
| healthy | 118 | 0 |

Bu sonuç, raw accuracy'nin neden yanıltıcı olduğunu doğrudan gösterir. Model bütün test pencerelerini `faulty` tahmin etmiştir. Test setinin çoğunluğu faulty olduğu için accuracy 0.8885 görünür, ancak healthy recall sıfırdır. Balanced accuracy 0.5000 ve macro F1 0.4705 bu başarısızlığı daha dürüst biçimde gösterir.

Anomaly-first analiz bu veri setinde daha anlamlıdır çünkü healthy session sayısı çok azdır ve supervised classifier healthy sınıfına genelleyememiştir. `results/kaist_baseline_experiment/tables/anomaly_metrics_table.md` dosyasına göre default reported anomaly baseline Isolation Forest olarak seçilmiştir:

| Model | Calibrated threshold | Balanced accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Isolation Forest | 0.6131 | 0.5268 | 0.8968 | 0.6468 | 0.7515 |

`results/kaist_baseline_experiment/tables/threshold_sweep_summary_table.md` dosyası threshold seçiminin sonucu ciddi etkilediğini gösterir. Isolation Forest için train-only calibrated threshold 0.6131 iken, held-out sweep içinde önerilen threshold 0.6230 ve balanced accuracy 0.7453 olarak raporlanmıştır. Bu ikinci değer optimistic analysis olarak kalmalıdır; train-only calibrated operating point ile karıştırılmamalıdır.

Split sensitivity bilgisi `results/kaist_baseline_experiment/publication_results_summary.md` dosyasında raporlanmıştır. Classifier balanced accuracy farklı healthy holdout splitlerinde 0.4851 ile 0.5000 aralığındadır. Bu bulgu, supervised classifier'ın mevcut session-safe koşullarda güvenilir olmadığını destekler.

## 6. KAIST RTF ve NASA IMS Anomaly-First Analizi

KAIST run-to-failure ve NASA IMS dense supervised classification veri seti gibi ele alınmamalıdır. Bu veri setlerinde snapshot/hour seviyesinde güvenilir dense target label bulunmadığı için label uydurmak yerine early healthy-reference calibration ve chronological anomaly scoring yaklaşımı kullanılmıştır.

KAIST run-to-failure sonuçları `results/kaist_rtf_experiment/experiment_summary.md`, `results/kaist_rtf_experiment/early_warning_summary.md` ve `results/kaist_rtf_experiment/threshold_crossing_summary.csv` dosyalarından alınmıştır:

- Dataset: `kaist_run_to_failure`.
- Analysis type: anomaly-first chronological degradation analysis.
- Signals used: vibration x/y, bearing temperature, ambient temperature.
- Source files: 129.
- Sessions: 1.
- Hourly feature rows: 129.
- Calibration region: ilk 24 saat, hour 0.0 to 23.0.
- Isolation Forest: threshold 0.711477, alarm yok.
- One-Class SVM: threshold 0.022606, first threshold crossing hour 24.0, first sustained warning hour 26.0.

Bu sonuçlar failure prediction olarak sunulmamalıdır. Doğru yorum: erken referans bölgesine göre davranış değişimi ve threshold crossing gözlemlenmiştir. Failure onset label olmadığı için alarm saati doğrulanmış arıza başlangıcı değildir.

NASA IMS sonuçları `results/nasa_ims_experiment/experiment_summary.md`, `results/nasa_ims_experiment/anomaly_model_summary.csv` ve `results/nasa_ims_experiment/threshold_crossing_summary.csv` dosyalarından alınmıştır:

- Dataset: `nasa_ims`.
- Analysis type: anomaly-first bearing-level degradation analysis.
- Signals used: vibration only.
- Discovered runs: 3.
- Bearing sessions: 12.
- Feature rows: 37856.
- Model features used: 44.
- Calibration: ilk 24 saat veya erken reference bölgesi; toplam reference files 1780.
- Isolation Forest: threshold 0.639746, sessions_with_alarm 4.
- One-Class SVM: threshold 2.244755, sessions_with_alarm 12.

NASA IMS için snapshot-level labels unknown olarak kalır. Documented end-of-run failures metadata olarak saklanmıştır ve dense target olarak kullanılmamıştır. Isolation Forest daha conservative alarm karakteristiği göstermektedir; One-Class SVM daha yaygın alarm üretmiştir. Bu fark tezde “alarm karakteristiği” olarak açıklanmalı, doğrudan model üstünlüğü olarak aşırı yorumlanmamalıdır.

## 7. Ana Riskler ve Önlemler

| Risk | Seviye | Neden | Tezde nasıl kontrol ediliyor? |
|---|---|---|---|
| Window-level leakage | Yüksek | Aynı session'dan türeyen pencerelerin train/test'e karışması performansı yapay yükseltebilir | KAIST rotating için session split; Paderborn için group split ve overlap kontrolü kullanılıyor |
| Class imbalance | Yüksek | KAIST rotating ve Paderborn'da faulty çoğunluk sınıfı baskın | Balanced accuracy, macro F1, confusion matrix ve train-only balancing raporlanıyor |
| Raw accuracy overclaim | Yüksek | KAIST rotating classifier tüm test pencerelerini faulty tahmin ederek 0.8885 accuracy alıyor | Accuracy headline yapılmıyor; balanced accuracy 0.5000 ve macro F1 0.4705 öne çıkarılıyor |
| Fabricated labels | Yüksek | KAIST RTF ve NASA IMS snapshot/hour seviyesinde dense supervised label sunmuyor | Bu veri setlerinde supervised classification yapılmıyor; anomaly-first yorum kullanılıyor |
| Fabricated synchronization | Orta | KAIST rotating modality'leri condition-matched ama güvenli time-synchronous değil | Raporlarda “condition-matched, unsynchronized” ifadesi korunuyor |
| Threshold sensitivity | Yüksek | Anomaly alarm sonuçları threshold seçimine duyarlı | Calibrated threshold ve optimistic sweep-selected threshold ayrı raporlanıyor |
| Acoustic vs AE confusion | Yüksek | KAIST acoustic verisi AE değildir | KAIST acoustic ayrı optional branch olarak yazılıyor; AE olarak adlandırılmıyor |
| Dataset scope creep | Orta | Her veri setine classification, anomaly, multiclass ve deep learning eklemek kapsamı dağıtabilir | İlk supervised benchmark Paderborn ile sınırlı; progression veri setleri anomaly-first tutuluyor |

## 8. Tezde Kullanılacak Ana Bulgular

1. Raw accuracy, ağır sınıf dengesizliği altında tek başına güvenilir değildir; KAIST rotating sonucu bunu açık biçimde göstermektedir.
2. Session/group-safe split, kestirimci bakım deneylerinde veri kaçağını önlemek için zorunludur.
3. Paderborn, mevcut altyapıda ilk temiz supervised benchmark olarak kullanılabilir; ancak sonuçlar balanced accuracy ve macro F1 üzerinden yorumlanmalıdır.
4. KAIST rotating, supervised classifier'ın sınıf dengesizliği altında healthy sınıfına genelleyemediği bir failure case sunmaktadır.
5. KAIST RTF ve NASA IMS, dense classification yerine anomaly-first ve degradation trend analizi için daha uygundur.
6. Threshold calibration, anomaly-first sonuçların yorumunu doğrudan etkiler; calibrated ve sweep-selected sonuçlar ayrı tutulmalıdır.
7. Classical ML baselineları, küçük veri ve tezde açıklanabilirlik gereksinimi için hâlâ uygundur.
8. Acoustic, AE ve time synchronization gibi semantik ayrımlar doğru yapılmadığında sonuçlar bilimsel olarak aşırı yorumlanabilir.

## 9. Tez/Sunum İçin Kısa Anlatım

Bu projede amacım, çoklu sensörlü kestirimci bakım için sadece yüksek accuracy veren bir demo değil, veri kaçağına dayanıklı ve tezde savunulabilir bir deney altyapısı kurmak. Bu nedenle değerlendirmeyi session veya group bazında ayırıyorum; aynı makine koşulundan veya aynı bearing code'dan gelen örneklerin train ve test arasında karışmasını engelliyorum. Smoke-test yalnızca pipeline'ın çalıştığını gösteriyor, gerçek benchmark olarak sunulmuyor. Paderborn veri seti şu anda ilk temiz supervised benchmark; vibration-only Random Forest ile balanced accuracy yaklaşık 0.592 ve macro F1 yaklaşık 0.594. KAIST rotating ise supervised classifier'ın ağır sınıf dengesizliği altında yanıltıcı raw accuracy üretebildiğini gösteriyor: accuracy yüksek görünse de model healthy sınıfını hiç yakalayamıyor. Bu yüzden ana tez çerçevesini anomaly-first kuruyorum. KAIST run-to-failure ve NASA IMS gibi progression veri setlerinde dense label uydurmadan, erken healthy-reference bölgesine göre sapma ve threshold crossing davranışını raporluyorum.

## 10. Sonraki Teknik İşler

1. Paderborn supervised benchmark için split-sensitivity analizi eklemek.
2. KAIST rotating için anomaly threshold calibration raporunu daha kısa ve tez tablosuna uygun hale getirmek.
3. KAIST RTF ve NASA IMS anomaly trend grafiklerini aynı görsel formatta karşılaştırmak.
4. CWRU için önce adapter specification yazmak; yerel dosya yapısı incelenmeden parser yazmamak.
5. Tüm veri setleri için tek sayfalık final benchmark table üretmek ve smoke-test sonuçlarını public benchmark sonuçlarından ayrı tutmak.
