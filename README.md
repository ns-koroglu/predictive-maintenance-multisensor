## Proje Başlığı

Çoklu Sensörlü Kestirimci Bakımda Anomaly-First ve Leakage-Aware Değerlendirme Altyapısı

## Öğrenci Bilgileri

- Öğrenci: Enes Köroğlu
- Öğrenci No: 1031120565
- Bölüm: Erciyes Üniversitesi, Mühendislik Fakültesi, Mekatronik Mühendisliği Bölümü
- Danışman: Dr. Şaban ULUS

## Kısa Amaç

Bu proje, çoklu sensörlü kestirimci bakım için erken arıza uyarısı ve anomali tespiti odaklı bir değerlendirme altyapısı sunar. Ana vurgu, public veri setleri üzerinde sınıf dengesizliği, veri kaçağı ve label belirsizliği risklerini açıkça yönetmektir. Çalışma production-ready endüstriyel sistem veya final RUL tahmini iddiası taşımaz.

## Teslim Klasörü Yapısı

| Klasör | İçerik |
|---|---|
| `report/` | MTU-I final Markdown raporu, uygunluk kontrol listesi ve mevcut advisor-facing DOCX çıktısı |
| `source/` | Proje kaynak kodu ve testler |
| `configs/` | Deney ve adapter YAML konfigürasyonları |
| `docs/` | Tezi destekleyen temiz teknik dokümantasyon ve adapter spesifikasyonları |
| `results/` | Raporu destekleyen kompakt metrikler, özetler ve grafikler |
| `figures/` | Raporda/sunumda kullanılabilecek seçilmiş görseller |

## Kurulum

Python sanal ortamı önerilir:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Bu temiz teslim paketinde kaynak kod `source/src` altında tutulur. Test veya script çalıştırırken `PYTHONPATH` ayarlanmalıdır:

```powershell
$env:PYTHONPATH = "source"
python -m pytest source/tests
```

## Ana Deney Komutları

Bu teslim klasörü raw public veri setlerini içermez. Aşağıdaki komutlar, tam repo ve ilgili veri klasörleri mevcut olduğunda çalıştırılabilir:

```powershell
python source/src/run_demo.py --config configs/experiment_config.yaml
python source/src/run_paderborn_experiment.py --config configs/paderborn_experiment.yaml
python source/src/run_kaist_experiment.py --config configs/kaist_experiment.yaml
python source/src/run_kaist_rtf_experiment.py --config configs/kaist_run_to_failure_experiment.yaml
python source/src/run_nasa_ims_experiment.py --config configs/nasa_ims_experiment.yaml
```

## Dahil Edilen Sonuçlar

- Paderborn binary vibration-only supervised benchmark metrikleri.
- KAIST rotating machine classifier failure ve anomaly-first analiz özetleri.
- KAIST run-to-failure anomaly trend ve threshold crossing özetleri.
- NASA IMS bearing-level anomaly progression özetleri.
- Smoke-test demo ve fusion/feature importance özetleri.

## Bilinçli Olarak Yapılmayan İddialar

- Bu çalışma production-ready endüstriyel sistem değildir.
- Final RUL tahmini yapılmamıştır.
- KAIST acoustic verisi AE olarak sunulmamıştır.
- KAIST run-to-failure ve NASA IMS supervised classification veri seti olarak sunulmamıştır.
- Public veri setleri MTU-I metodoloji doğrulaması için kullanılmıştır.
- Fiziksel AE/vibration/thermal senkron laboratuvar verisi gelecek çalışma olarak planlanmıştır.
