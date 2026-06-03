# Proje Manifesti

| Path | Purpose | Included because | Notes |
|---|---|---|---|
| `README.md` | Teslim paketi açıklaması | Danışman ve okul için hızlı yönlendirme sağlar | Temiz ve advisor-facing olarak yeniden yazıldı |
| `requirements.txt` | Python bağımlılıkları | Projenin kurulabilirliğini destekler | Root dosyadan kopyalandı |
| `report/Enes_Koroglu_1031120565_MTU1_Rapor.docx` | Resmi MTU-I raporu | Okul/danışman teslimi için final Word raporu | İç proje çıktıları bölümü kaldırılarak temizlendi |
| `report/Enes_Koroglu_1031120565_MTU1_Rapor.pdf` | Resmi MTU-I raporu | Okul/danışman teslimi için final PDF raporu | Temizlenmiş DOCX dosyasından yeniden üretildi |
| `report/Enes_Koroglu_1031120565_MTU1_Degerlendirme_Formu.docx` | MTU-I değerlendirme formu | Resmi değerlendirme evrakı | İçerik değiştirilmeden kopyalandı |
| `report/Enes_Koroglu_1031120565_MTU1_Degerlendirme_Formu.pdf` | MTU-I değerlendirme formu | Resmi değerlendirme evrakı | İçerik değiştirilmeden kopyalandı |
| `report/mtu1_final_report_submission_tr.md` | Final MTU-I raporu | Ana akademik teslim dokümanı | İç proje çıktıları bölümü kaldırılmış final Markdown sürüm |
| `report/mtu1_report_compliance_checklist_tr.md` | MTU-I uygunluk kontrolü | Değerlendirme kriterlerini izler | Final raporu destekler |
| `report/hoca_sunum_paketi_tr.docx` | Mevcut advisor-facing DOCX paket | Danışman görüşmesi için ek sunum/rapor paketi | Resmi rapor/form dosyalarına ek opsiyonel çıktı; manuel kontrol önerilir |
| `source/src/` | Kaynak kod | Pipeline, adapterler ve deney runnerları burada | Cache ve bytecode dosyaları hariç |
| `source/tests/` | Testler | Yazılımın doğrulanabilirliğini gösterir | Cache dosyaları hariç |
| `configs/` | YAML konfigürasyonları | Reproducible deney akışını destekler | Tüm `.yaml` dosyaları dahil |
| `docs/benchmark_status_summary_tr.md` | Benchmark özeti | Rapor bulgularını destekler | Temiz teknik özet |
| `docs/tez_bulgular_ve_tartisma_taslagi_tr.md` | Bulgular/tartışma taslağı | Final raporun teknik arka planını destekler | Teslimde ek doküman olarak yararlı |
| `docs/multi_dataset_framework.md` | Framework açıklaması | Multi-dataset yapı için teknik kanıt | İçerik kısa ve temiz |
| `docs/kaist_adapter_spec.md` | KAIST adapter spesifikasyonu | Veri semantiği ve sınırları açıklar | Acoustic ≠ AE ayrımı için önemli |
| `docs/nasa_ims_adapter_spec.md` | NASA IMS adapter spesifikasyonu | Dense label olmayan progression şemasını açıklar | Anomaly-first gerekçesini destekler |
| `docs/paderborn_adapter_spec.md` | Paderborn adapter spesifikasyonu | Supervised benchmark semantiğini açıklar | Group-aware split için önemli |
| `docs/project_brief.md` | Proje kapsamı | MTU-I hedefini özetler | Clean supporting document |
| `docs/thesis_outline.md` | Tez taslak yapısı | Devam planını gösterir | Kısa destek dokümanı |
| `docs/data_schema.md` | Veri şeması | Session formatını açıklar | Teknik ek niteliğinde |
| `docs/literature_notes.md` | Literatür temaları | Literatür yönelimini gösterir | Ayrıntılı kaynakça yerine final rapordaki akademik kaynakça esas alınmalıdır |
| `results/raw_smoke_real_data_readiness/` | Smoke-test özet sonuçları | Pipeline doğrulamasını destekler | Gerçek benchmark olarak sunulmamalı |
| `results/paderborn_baseline_experiment/` | Paderborn kompakt kanıtları | İlk temiz supervised benchmarkı destekler | Model binary ve büyük ara çıktılar hariç |
| `results/kaist_baseline_experiment/` | KAIST rotating kompakt kanıtları | Raw accuracy/class imbalance tartışmasını destekler | Büyük datasets, predictions ve model binaryler hariç |
| `results/kaist_rtf_experiment/` | KAIST RTF kompakt kanıtları | Anomaly-first progression analizini destekler | CSV trendin büyük versiyonu hariç |
| `results/nasa_ims_experiment/` | NASA IMS kompakt kanıtları | Bearing-level anomaly progression analizini destekler | 34 MB anomaly trend CSV hariç |
| `figures/` | Seçilmiş görseller | Rapor ve sunumda kullanılacak ana grafikler | Karmaşık/tekrarlı figürler azaltıldı |
