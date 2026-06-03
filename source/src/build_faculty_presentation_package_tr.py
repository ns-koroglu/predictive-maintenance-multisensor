"""Build a Turkish faculty presentation package from existing experiment outputs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "docs" / "faculty_presentation_package_tr"
FIGURES_DIR = PACKAGE_DIR / "figures"
TABLES_DIR = PACKAGE_DIR / "tables"


@dataclass
class PackageContext:
    smoke_classifier: dict[str, Any]
    smoke_anomaly: dict[str, Any]
    smoke_ablation: pd.DataFrame
    smoke_sensor_importance: pd.DataFrame
    smoke_top_features: pd.DataFrame
    smoke_sessions: pd.DataFrame
    kaist_classifier: dict[str, Any]
    kaist_if: dict[str, Any]
    kaist_ocsvm: dict[str, Any]
    kaist_threshold_sweep: pd.DataFrame
    kaist_anomaly_comparison: pd.DataFrame
    kaist_split_summary: pd.DataFrame
    kaist_sessions_manifest: pd.DataFrame
    kaist_modality_manifest: pd.DataFrame
    kaist_feature_dataset_rows: int
    kaist_train_windows: pd.DataFrame
    kaist_test_windows: pd.DataFrame
    kaist_rtf_summary: dict[str, Any]
    kaist_rtf_trend: pd.DataFrame
    kaist_rtf_threshold: pd.DataFrame
    kaist_rtf_model_summary: pd.DataFrame
    nasa_adapter_summary: dict[str, Any]
    nasa_run_manifest: pd.DataFrame
    nasa_session_manifest: pd.DataFrame
    nasa_feature_rows: int
    nasa_summary: dict[str, Any]
    nasa_trend: pd.DataFrame
    nasa_threshold: pd.DataFrame
    nasa_model_summary: pd.DataFrame


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dirs() -> None:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _read_context() -> PackageContext:
    smoke_root = ROOT / "results" / "raw_smoke_real_data_readiness"
    kaist_root = ROOT / "results" / "kaist_baseline_experiment"
    kaist_rtf_root = ROOT / "results" / "kaist_rtf_experiment"
    nasa_root = ROOT / "results" / "nasa_ims_experiment"
    kaist_processed = ROOT / "data" / "processed" / "kaist_rotating_machine"
    kaist_rtf_processed = ROOT / "data" / "processed" / "kaist_run_to_failure"
    nasa_processed = ROOT / "data" / "processed" / "nasa_ims"
    kaist_feature_dataset = ROOT / "results" / "kaist_feature_build" / "datasets" / "kaist_vibration_thermal_features.csv"
    nasa_feature_dataset = nasa_processed / "datasets" / "nasa_ims_bearing_feature_dataset.csv"
    return PackageContext(
        smoke_classifier=_load_json(smoke_root / "metrics" / "classifier_metrics.json"),
        smoke_anomaly=_load_json(smoke_root / "metrics" / "anomaly_metrics.json"),
        smoke_ablation=pd.read_csv(smoke_root / "ablation_comparison.csv"),
        smoke_sensor_importance=pd.read_csv(smoke_root / "sensor_group_importance.csv"),
        smoke_top_features=pd.read_csv(smoke_root / "top_features.csv"),
        smoke_sessions=pd.read_csv(smoke_root / "demo_session_results.csv"),
        kaist_classifier=_load_json(kaist_root / "metrics" / "classifier_metrics.json"),
        kaist_if=_load_json(kaist_root / "metrics" / "isolation_forest_metrics.json"),
        kaist_ocsvm=_load_json(kaist_root / "metrics" / "one_class_svm_metrics.json"),
        kaist_threshold_sweep=pd.read_csv(kaist_root / "threshold_sweep.csv"),
        kaist_anomaly_comparison=pd.read_csv(kaist_root / "anomaly_baseline_comparison.csv"),
        kaist_split_summary=pd.read_csv(kaist_root / "tables" / "split_sensitivity_summary.csv"),
        kaist_sessions_manifest=pd.read_csv(kaist_processed / "manifests" / "sessions_manifest.csv"),
        kaist_modality_manifest=pd.read_csv(kaist_processed / "manifests" / "modality_availability.csv"),
        kaist_feature_dataset_rows=sum(1 for _ in kaist_feature_dataset.open("r", encoding="utf-8")) - 1,
        kaist_train_windows=pd.read_csv(kaist_root / "datasets" / "train_windows.csv", usecols=["label"]),
        kaist_test_windows=pd.read_csv(kaist_root / "datasets" / "test_windows.csv", usecols=["label"]),
        kaist_rtf_summary=_load_json(kaist_rtf_root / "experiment_summary.json"),
        kaist_rtf_trend=pd.read_csv(kaist_rtf_root / "anomaly_trend.csv"),
        kaist_rtf_threshold=pd.read_csv(kaist_rtf_root / "threshold_crossing_summary.csv"),
        kaist_rtf_model_summary=pd.read_csv(kaist_rtf_root / "anomaly_model_summary.csv"),
        nasa_adapter_summary=_load_json(nasa_processed / "adapter_summary.json"),
        nasa_run_manifest=pd.read_csv(nasa_processed / "manifests" / "run_manifest.csv"),
        nasa_session_manifest=pd.read_csv(nasa_processed / "manifests" / "bearing_session_manifest.csv"),
        nasa_feature_rows=sum(1 for _ in nasa_feature_dataset.open("r", encoding="utf-8")) - 1,
        nasa_summary=_load_json(nasa_root / "experiment_summary.json"),
        nasa_trend=pd.read_csv(nasa_root / "anomaly_trend.csv", low_memory=False),
        nasa_threshold=pd.read_csv(nasa_root / "threshold_crossing_summary.csv", low_memory=False),
        nasa_model_summary=pd.read_csv(nasa_root / "anomaly_model_summary.csv"),
    )


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _df_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [[_fmt(value, 4) for value in row] for row in df.itertuples(index=False, name=None)]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])


def _save_table(df: pd.DataFrame, stem: str) -> None:
    df.to_csv(TABLES_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig")
    (TABLES_DIR / f"{stem}.md").write_text(_df_to_markdown(df), encoding="utf-8")


def _save_json(data: Any, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_tables(ctx: PackageContext) -> None:
    smoke_total_windows = int(ctx.smoke_sessions["n_windows"].fillna(0).sum())
    kaist_primary_sessions = int((ctx.kaist_sessions_manifest["session_branch"] == "primary").sum())
    kaist_optional_acoustic_sessions = int((ctx.kaist_sessions_manifest["session_branch"] == "optional_acoustic").sum())

    dataset_table = pd.DataFrame(
        [
            {
                "veri_seti": "Smoke-test demo",
                "tur": "Sentetik / şema doğrulama",
                "modaliteler": "AE + titreşim + termal",
                "oturum_veya_kosul": "4 oturum",
                "ozellik_satiri_pencere": smoke_total_windows,
                "etiket_yapisi": "Yoğun ve tam bilinen sentetik etiket",
                "ana_kullanim": "Pipeline ve raporlama doğrulaması",
                "temel_kisit": "Gerçek genelleme kanıtı değildir.",
            },
            {
                "veri_seti": "KAIST Rotating Machine",
                "tur": "Koşul-eşlemeli kamu veri seti",
                "modaliteler": "Titreşim + termal, akım korunuyor, akustik ayrı",
                "oturum_veya_kosul": f"{kaist_primary_sessions} birincil oturum, {kaist_optional_acoustic_sessions} opsiyonel akustik oturum",
                "ozellik_satiri_pencere": ctx.kaist_feature_dataset_rows,
                "etiket_yapisi": "İkili healthy/faulty + çok sınıflı arıza ailesi",
                "ana_kullanim": "Session-safe sınıflandırma ve anomaly-first karşılaştırma",
                "temel_kisit": "Condition-matched; güvenli zaman senkronu yok. Acoustic AE değildir.",
            },
            {
                "veri_seti": "KAIST Run-to-Failure",
                "tur": "Saatlik ilerleme dizisi",
                "modaliteler": "Titreşim + rulman sıcaklığı + ortam sıcaklığı",
                "oturum_veya_kosul": "1 run-to-failure oturumu",
                "ozellik_satiri_pencere": ctx.kaist_rtf_summary["n_feature_rows"],
                "etiket_yapisi": "Yoğun etiket yok; anomaly-first referans/evaluation ayrımı",
                "ana_kullanim": "Erken uyarı ve eşik davranışı",
                "temel_kisit": "Tek koşu olduğu için varyans analizi sınırlı.",
            },
            {
                "veri_seti": "NASA IMS",
                "tur": "Run-to-failure rulman veri seti",
                "modaliteler": "Titreşim",
                "oturum_veya_kosul": f"{ctx.nasa_adapter_summary['n_runs']} run, {ctx.nasa_adapter_summary['n_bearing_sessions']} bearing session",
                "ozellik_satiri_pencere": ctx.nasa_feature_rows,
                "etiket_yapisi": "Snapshot düzeyinde unknown; run sonu failure metadata ayrı tutulur",
                "ana_kullanim": "Anomali trendi ve erken alarm davranışı",
                "temel_kisit": "Nested 3rd/4th path semantik belirsizlik içerir.",
            },
        ]
    )
    _save_table(dataset_table, "01_veri_seti_karsilastirma")

    model_table = pd.DataFrame(
        [
            {
                "model": "Random Forest",
                "rol": "Denetlenmiş temel sınıflandırıcı",
                "egitim_rejimi": "Class-weight + isteğe bağlı train-only balancing",
                "kullanildigi_yer": "Smoke-test ve KAIST rotating",
                "neden_secildi": "Açıklanabilir, küçük veri altında güçlü, özellik önemleri üretir",
                "sinir": "Ağır sınıf dengesizliğinde majority sınıfa çökebilir.",
            },
            {
                "model": "Isolation Forest",
                "rol": "Varsayılan anomaly-first baseline",
                "egitim_rejimi": "Sağlıklı referans bölgesi ile kalibrasyon",
                "kullanildigi_yer": "Smoke-test, KAIST rotating, KAIST RTF, NASA IMS",
                "neden_secildi": "Basit, hızlı ve konservatif alarm karakteristiği",
                "sinir": "Eşik seçimi sonuca güçlü biçimde etki eder.",
            },
            {
                "model": "One-Class SVM",
                "rol": "İkinci anomaly baseline",
                "egitim_rejimi": "Sağlıklı referans bölgesi ile kalibrasyon",
                "kullanildigi_yer": "KAIST rotating, KAIST RTF, NASA IMS",
                "neden_secildi": "Sağlıklı manifold duyarlılığı ve erken alarm potansiyeli",
                "sinir": "Agresif alarm eğilimi yüksek false positive riski doğurabilir.",
            },
        ]
    )
    _save_table(model_table, "02_kullanilan_modeller")

    metrics_table = pd.DataFrame(
        [
            {
                "metrik": "Raw accuracy",
                "ne_olcer": "Toplam doğru tahmin oranı",
                "ne_zaman_kullanildi": "Raporlandı ama ana karar metriği yapılmadı",
                "neden_tek_basina_yetersiz": "Sınıf dengesizliğinde healthy sınıf tamamen kaçırılsa bile yüksek görünebilir.",
            },
            {
                "metrik": "Balanced accuracy",
                "ne_olcer": "Sınıf başına recall ortalaması",
                "ne_zaman_kullanildi": "KAIST rotating ana dürüstlük metriği",
                "neden_tek_basina_yetersiz": "Karşılaştırma için güçlüdür ama alarm zamanlamasını açıklamaz.",
            },
            {
                "metrik": "Macro precision / recall / F1",
                "ne_olcer": "Sınıflara eşit ağırlıklı sınıflandırma başarımı",
                "ne_zaman_kullanildi": "Dengesiz ikili sınıflandırma için",
                "neden_tek_basina_yetersiz": "Chronological trend ve alarm sürekliliğini göstermez.",
            },
            {
                "metrik": "ROC-AUC / PR-AUC",
                "ne_olcer": "Skor ayrıştırma gücü",
                "ne_zaman_kullanildi": "KAIST rotating karşılaştırmaları",
                "neden_tek_basina_yetersiz": "Seçilen gerçek eşik davranışını tek başına açıklamaz.",
            },
            {
                "metrik": "Threshold crossing / first alarm hour",
                "ne_olcer": "İlk alarm ve sürdürülen uyarı zamanı",
                "ne_zaman_kullanildi": "KAIST RTF ve NASA IMS anomaly-first deneyleri",
                "neden_tek_basina_yetersiz": "Alarmın doğru veya faydalı olduğu sonucu için ek bağlam gerekir.",
            },
        ]
    )
    _save_table(metrics_table, "03_degerlendirme_metrikleri")

    kaist_rotating_table = pd.DataFrame(
        [
            {
                "yaklasim": "Random Forest classifier",
                "raporlama_tipi": "Calibrated / varsayılan",
                "accuracy": ctx.kaist_classifier["accuracy"],
                "balanced_accuracy": ctx.kaist_classifier["balanced_accuracy"],
                "precision": ctx.kaist_classifier["precision_macro"],
                "recall": ctx.kaist_classifier["recall_macro"],
                "f1": ctx.kaist_classifier["f1_macro"],
                "pr_auc": ctx.kaist_classifier["pr_auc"],
                "yorum": "Tüm test pencerelerini faulty tahmin etti; healthy genellemesi yok.",
            },
            {
                "yaklasim": "Isolation Forest",
                "raporlama_tipi": "Calibrated / varsayılan",
                "accuracy": ctx.kaist_if["accuracy"],
                "balanced_accuracy": ctx.kaist_if["balanced_accuracy"],
                "precision": ctx.kaist_if["precision"],
                "recall": ctx.kaist_if["recall"],
                "f1": ctx.kaist_if["f1"],
                "pr_auc": ctx.kaist_if["pr_auc"],
                "yorum": "Daha dengeli ve savunulabilir anomaly-first temel sonuç.",
            },
            {
                "yaklasim": "Isolation Forest",
                "raporlama_tipi": "Threshold sweep en iyi nokta",
                "accuracy": np.nan,
                "balanced_accuracy": ctx.kaist_if["threshold_analysis"]["recommended_balanced_accuracy"],
                "precision": ctx.kaist_if["threshold_analysis"]["recommended_precision"],
                "recall": ctx.kaist_if["threshold_analysis"]["recommended_recall"],
                "f1": ctx.kaist_if["threshold_analysis"]["recommended_f1"],
                "pr_auc": np.nan,
                "yorum": "İyimser eşik; doğrudan headline sonuç yapılmadı.",
            },
            {
                "yaklasim": "One-Class SVM",
                "raporlama_tipi": "Calibrated / varsayılan",
                "accuracy": ctx.kaist_ocsvm["accuracy"],
                "balanced_accuracy": ctx.kaist_ocsvm["balanced_accuracy"],
                "precision": ctx.kaist_ocsvm["precision"],
                "recall": ctx.kaist_ocsvm["recall"],
                "f1": ctx.kaist_ocsvm["f1"],
                "pr_auc": ctx.kaist_ocsvm["pr_auc"],
                "yorum": "Agresif alarm; tüm pencereleri anomaly sayma eğilimi gösterdi.",
            },
            {
                "yaklasim": "One-Class SVM",
                "raporlama_tipi": "Threshold sweep en iyi nokta",
                "accuracy": np.nan,
                "balanced_accuracy": ctx.kaist_ocsvm["threshold_analysis"]["recommended_balanced_accuracy"],
                "precision": ctx.kaist_ocsvm["threshold_analysis"]["recommended_precision"],
                "recall": ctx.kaist_ocsvm["threshold_analysis"]["recommended_recall"],
                "f1": ctx.kaist_ocsvm["threshold_analysis"]["recommended_f1"],
                "pr_auc": np.nan,
                "yorum": "Yalnızca iyimser eşik analizi olarak değerlendirildi.",
            },
        ]
    )
    _save_table(kaist_rotating_table, "04_kaist_rotating_temas_sonuclar")

    kaist_rtf_table = pd.DataFrame(
        [
            {
                "model": row["model_name"],
                "kalibrasyon": "İlk 24 saat",
                "kalibrasyon_ornek_sayisi": row["n_reference_files"],
                "kalibre_esik": row["calibrated_threshold"],
                "ilk_alarm_saati": row["first_alarm_hour"],
                "ilk_alarm_nedeni": row["first_alarm_reason"],
                "durum": row["status"],
                "yorum": (
                    "Konservatif; değerlendirme bölgesinde alarm üretmedi."
                    if row["model_name"] == "isolation_forest"
                    else "Agresif; kalibrasyon sonrasında hızlı ve sürekli alarm verdi."
                ),
            }
            for _, row in ctx.kaist_rtf_model_summary.iterrows()
        ]
    )
    _save_table(kaist_rtf_table, "05_kaist_rtf_temas_sonuclar")

    nasa_failed = ctx.nasa_threshold[
        ctx.nasa_threshold["documented_failed_bearing"].fillna(False).astype(bool)
    ]
    nasa_basic_rows: list[dict[str, Any]] = []
    for _, row in ctx.nasa_model_summary.iterrows():
        model_name = row["model_name"]
        failed_subset = nasa_failed[nasa_failed["model_name"] == model_name]
        nasa_basic_rows.append(
            {
                "model": model_name,
                "kalibrasyon": "Her bearing için ilk 24 saat",
                "kalibre_esik": row["calibrated_threshold"],
                "alarm_veren_session_sayisi": int(row["sessions_with_alarm"]),
                "dokumante_arizali_bearing_alarm_orani": f"{int((failed_subset['status'] == 'alarm_detected').sum())}/{len(failed_subset)}",
                "nested_path_notu": "3rd_test/4th_test/txt korunarak işlendi",
                "yorum": (
                    "Daha seçici; yalnızca 4/12 bearing session alarm verdi."
                    if model_name == "isolation_forest"
                    else "Çok agresif; 12/12 session alarm verdi."
                ),
            }
        )
    _save_table(pd.DataFrame(nasa_basic_rows), "06_nasa_ims_temas_sonuclar")

    anomaly_comparison_rows = [
        {
            "veri_seti": "KAIST Rotating",
            "model": "Isolation Forest",
            "alarm_karakteristigi": "Daha konservatif",
            "temel_bulgusu": "Calibrated balanced accuracy 0.5268, F1 0.7515",
            "yorum": "Varsayılan raporlanan anomaly baseline.",
        },
        {
            "veri_seti": "KAIST Rotating",
            "model": "One-Class SVM",
            "alarm_karakteristigi": "Çok agresif",
            "temel_bulgusu": "Calibrated balanced accuracy 0.5000, F1 0.9409",
            "yorum": "Yüksek F1, dengesiz veri nedeniyle yanıltıcı olabilir.",
        },
        {
            "veri_seti": "KAIST RTF",
            "model": "Isolation Forest",
            "alarm_karakteristigi": "Konservatif",
            "temel_bulgusu": "129 saatlik trendde alarm vermedi",
            "yorum": "False negative riski taşıyan güvenli davranış.",
        },
        {
            "veri_seti": "KAIST RTF",
            "model": "One-Class SVM",
            "alarm_karakteristigi": "Erken ve sürekli",
            "temel_bulgusu": "İlk kalıcı alarm 26. saatte",
            "yorum": "Erken uyarı potansiyeli var, false positive riski yüksek.",
        },
        {
            "veri_seti": "NASA IMS",
            "model": "Isolation Forest",
            "alarm_karakteristigi": "Seçici",
            "temel_bulgusu": "4/12 session alarm, dokümante arızalı 2 bearing yakalandı",
            "yorum": "Kısmi duyarlılık, sınırlı false positive kontrolü.",
        },
        {
            "veri_seti": "NASA IMS",
            "model": "One-Class SVM",
            "alarm_karakteristigi": "Agresif",
            "temel_bulgusu": "12/12 session alarm",
            "yorum": "Alarm karakteristiği çok erken ve geniş.",
        },
    ]
    _save_table(pd.DataFrame(anomaly_comparison_rows), "07_anomali_model_karsilastirma")

    roadmap_table = pd.DataFrame(
        [
            {
                "donem": "Kısa vade",
                "zaman": "0-2 ay",
                "hedef": "Gerçek veri toplama düzeneğinin kesinleştirilmesi ve ilk pilot ölçüm",
                "cikti": "Sensör yerleşimi, veri şeması, pilot oturumlar",
                "risk": "Düzeneğin gecikmesi",
                "yedek_plan": "Kamu veri seti benchmark hattını derinleştirerek tez omurgasını korumak",
            },
            {
                "donem": "Kısa vade",
                "zaman": "0-2 ay",
                "hedef": "KAIST rotating ve NASA IMS için daha güçlü split-sensitivity ve threshold kararlılığı analizi",
                "cikti": "Ek tablo ve yayın seviyesi ek deneyler",
                "risk": "Healthy örnek sayısının yetersizliği",
                "yedek_plan": "Anomaly-first hattı ana tez ekseni yapmak",
            },
            {
                "donem": "Orta vade",
                "zaman": "2-4 ay",
                "hedef": "Paderborn ve/veya CWRU kompakt adapter entegrasyonu",
                "cikti": "Ek benchmark veri setleri",
                "risk": "Semantik uyumsuzluk",
                "yedek_plan": "Sadece NASA IMS + KAIST ailesi ile savunulabilir benchmark seti kurmak",
            },
            {
                "donem": "Orta vade",
                "zaman": "2-4 ay",
                "hedef": "Gerçek deney verisi ile session-safe ilk iç benchmarkın kurulması",
                "cikti": "Gerçek healthy ve gelişen arıza oturumları",
                "risk": "Yoğun etiket üretiminin zor olması",
                "yedek_plan": "Erken uyarı ve anomaly trend üzerine odaklanmak",
            },
            {
                "donem": "Uzun vade",
                "zaman": "4-8 ay",
                "hedef": "Yayın hedefi için karşılaştırmalı deney setinin tamamlanması",
                "cikti": "Makale taslağı, sonuç tabloları, sınırlılık analizi",
                "risk": "Sonuçların heterojen kalması",
                "yedek_plan": "Negatif sonuçları dürüstçe sınırlılık ve tasarım kararı olarak çerçevelemek",
            },
        ]
    )
    _save_table(roadmap_table, "08_gelecek_adimlar_takvimi")


def _annotate_bars(ax: plt.Axes, bars: Any, fmt_digits: int = 2) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01 * max(1.0, height),
            f"{height:.{fmt_digits}f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def _save_fig(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, facecolor: str) -> None:
    patch = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor="#334155",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=12, wrap=True)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=2, color="#334155"))


def _figure_system_architecture() -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")
    ax.set_title("Proje Mimarisi ve Deney Akışı", fontsize=20, weight="bold")
    _draw_box(ax, (0.04, 0.62), 0.2, 0.2, "Veri girişi\nSmoke-test / Kamu veri setleri /\nGerçek oturumlar", "#DBEAFE")
    _draw_box(ax, (0.28, 0.62), 0.2, 0.2, "Adapter / veri yükleme\nŞema doğrulama\nSemantik kısıtların korunması", "#E0F2FE")
    _draw_box(ax, (0.52, 0.62), 0.2, 0.2, "Ön işleme ve pencereleme\nSession-safe mantık\nOrtak deney şeması", "#DCFCE7")
    _draw_box(ax, (0.76, 0.62), 0.2, 0.2, "Özellik çıkarımı ve fusion\nAçıklanabilir klasik özellikler", "#FEF3C7")
    _draw_box(ax, (0.22, 0.25), 0.24, 0.2, "Random Forest\nSınıflandırma baseline\nSadece uygun veri setlerinde", "#FCE7F3")
    _draw_box(ax, (0.54, 0.25), 0.24, 0.2, "Isolation Forest / OCSVM\nAnomaly-first ve erken uyarı\nKalibrasyon bölgesi + eşik analizi", "#F3E8FF")
    _draw_box(ax, (0.80, 0.25), 0.16, 0.2, "Raporlama\nGrafikler\nTablolar\nMarkdown özetleri", "#FEE2E2")
    _arrow(ax, (0.24, 0.72), (0.28, 0.72))
    _arrow(ax, (0.48, 0.72), (0.52, 0.72))
    _arrow(ax, (0.72, 0.72), (0.76, 0.72))
    _arrow(ax, (0.64, 0.62), (0.62, 0.45))
    _arrow(ax, (0.64, 0.62), (0.66, 0.45))
    _arrow(ax, (0.46, 0.35), (0.54, 0.35))
    _arrow(ax, (0.78, 0.35), (0.80, 0.35))
    ax.text(
        0.5,
        0.05,
        "Ana ilke: veri seti semantiğini bozmadan, session-safe ve anomaly-first değerlendirme yapmak.",
        ha="center",
        fontsize=12,
    )
    _save_fig(fig, "01_sistem_mimarisi_ve_deney_akisi.png")


def _figure_smoke_test_summary(ctx: PackageContext) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Smoke-test Demo Özeti", fontsize=20, weight="bold")

    counts = (
        ctx.smoke_sessions.groupby("true_label")["n_windows"].sum().reindex(["healthy", "developing_fault"]).fillna(0)
    )
    bars = axes[0, 0].bar(["Healthy", "Gelişen arıza"], counts.values, color=["#16A34A", "#DC2626"])
    axes[0, 0].set_title("Pencere sınıf dağılımı")
    axes[0, 0].set_ylabel("Pencere sayısı")
    _style_axes(axes[0, 0])
    _annotate_bars(axes[0, 0], bars, fmt_digits=0)

    metric_names = ["Classifier Macro F1", "Anomaly F1", "Anomaly Recall"]
    metric_values = [
        ctx.smoke_classifier["f1_macro"],
        ctx.smoke_anomaly["f1"],
        ctx.smoke_anomaly["recall"],
    ]
    bars = axes[0, 1].bar(metric_names, metric_values, color=["#2563EB", "#F97316", "#F59E0B"])
    axes[0, 1].set_ylim(0, 1.1)
    axes[0, 1].set_title("Ana smoke-test metrikleri")
    axes[0, 1].tick_params(axis="x", rotation=15)
    _style_axes(axes[0, 1])
    _annotate_bars(axes[0, 1], bars)

    ablation = ctx.smoke_ablation.set_index("setup_name").loc[
        ["ae_only", "vibration_only", "thermal_only", "fused"]
    ]
    bars = axes[1, 0].bar(
        ["AE", "Titreşim", "Termal", "Fusion"],
        ablation["anomaly_f1"].values,
        color=["#60A5FA", "#2563EB", "#F97316", "#10B981"],
    )
    axes[1, 0].set_ylim(0, 1.0)
    axes[1, 0].set_title("Ablation: anomaly F1")
    _style_axes(axes[1, 0])
    _annotate_bars(axes[1, 0], bars)

    inference_row = ctx.smoke_sessions[ctx.smoke_sessions["demo_role"] == "inference_target"].iloc[0]
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Split ve çıkarım özeti")
    summary_text = (
        "Train oturumları:\n"
        "- session_002_healthy\n"
        "- session_003_developing_fault\n\n"
        "Test oturumları:\n"
        "- session_001_healthy\n"
        "- session_004_developing_fault\n\n"
        f"Çıkarım hedefi: {inference_row['session_id']}\n"
        f"Tahmin: {inference_row['inference_predicted_class']}\n"
        f"Erken uyarı: {str(bool(inference_row['inference_early_warning']))}\n"
        f"Neden: {inference_row['warning_trigger_reason']}"
    )
    axes[1, 1].text(
        0.02,
        0.95,
        summary_text,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
    )
    _save_fig(fig, "02_smoke_test_demo_ozeti.png")


def _figure_kaist_classifier_failure(ctx: PackageContext) -> None:
    train_counts = ctx.kaist_train_windows["label"].value_counts()
    test_counts = ctx.kaist_test_windows["label"].value_counts()
    total_counts = (train_counts.add(test_counts, fill_value=0)).reindex(["faulty", "healthy"]).fillna(0)
    confusion = np.array(ctx.kaist_classifier["confusion_matrix"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("KAIST Rotating: Sınıf Dengesizliği ve Sınıflandırıcı Hatası", fontsize=18, weight="bold")

    bars = axes[0].bar(["Faulty", "Healthy"], total_counts.values, color=["#2563EB", "#DC2626"])
    axes[0].set_title("Toplam pencere dağılımı")
    axes[0].set_ylabel("Pencere sayısı")
    _style_axes(axes[0])
    _annotate_bars(axes[0], bars, fmt_digits=0)
    axes[0].text(
        0.02,
        0.92,
        "Raw accuracy bu dağılım altında tek başına\nbaşarı göstergesi değildir.",
        transform=axes[0].transAxes,
        fontsize=11,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF", edgecolor="#93C5FD"),
    )

    im = axes[1].imshow(confusion, cmap="Blues")
    axes[1].set_xticks([0, 1], labels=["Faulty", "Healthy"])
    axes[1].set_yticks([0, 1], labels=["Faulty", "Healthy"])
    axes[1].set_xlabel("Tahmin")
    axes[1].set_ylabel("Gerçek")
    axes[1].set_title("Test confusion matrix")
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            axes[1].text(j, i, str(confusion[i, j]), ha="center", va="center", fontsize=13)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].text(
        0.03,
        -0.20,
        f"Balanced accuracy = {ctx.kaist_classifier['balanced_accuracy']:.4f}, "
        f"Macro F1 = {ctx.kaist_classifier['f1_macro']:.4f}",
        transform=axes[1].transAxes,
        fontsize=11,
    )
    _save_fig(fig, "03_kaist_rotating_siniflandirici_basarisizligi_ve_sinif_dengesizligi.png")


def _figure_kaist_anomaly_comparison(ctx: PackageContext) -> None:
    df = ctx.kaist_anomaly_comparison.copy()
    metrics = ["balanced_accuracy", "precision", "recall", "f1"]
    labels = ["Balanced\nAccuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title("KAIST Rotating: Kalibre Anomali Baseline Karşılaştırması", fontsize=18, weight="bold")
    if_row = df[df["model_name"] == "isolation_forest"].iloc[0]
    oc_row = df[df["model_name"] == "one_class_svm"].iloc[0]
    bars1 = ax.bar(x - width / 2, [if_row[m] for m in metrics], width, label="Isolation Forest", color="#2563EB")
    bars2 = ax.bar(x + width / 2, [oc_row[m] for m in metrics], width, label="One-Class SVM", color="#F97316")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    _style_axes(ax)
    _annotate_bars(ax, bars1)
    _annotate_bars(ax, bars2)
    ax.text(
        0.02,
        0.02,
        "Not: OCSVM'nin yüksek F1 değeri, tüm pencereleri anomaly saymaya yaklaşan agresif alarm karakteristiği ile birlikte okunmalıdır.",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF7ED", edgecolor="#FDBA74"),
    )
    _save_fig(fig, "04_kaist_rotating_anomali_baseline_karsilastirmasi.png")


def _figure_kaist_threshold_sweep(ctx: PackageContext) -> None:
    df = ctx.kaist_threshold_sweep.copy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("KAIST Rotating: Threshold Sweep Analizi", fontsize=18, weight="bold")

    colors = {"isolation_forest": "#2563EB", "one_class_svm": "#F97316"}
    for model_name, subset in df.groupby("model_name"):
        subset = subset.sort_values("threshold")
        axes[0].plot(subset["threshold"], subset["precision"], color=colors[model_name], label=f"{model_name} precision")
        axes[0].plot(subset["threshold"], subset["recall"], color=colors[model_name], linestyle="--", label=f"{model_name} recall")
        axes[1].plot(subset["threshold"], subset["f1"], color=colors[model_name], label=f"{model_name} F1")
        calibrated = subset[subset["is_calibrated_threshold"]]
        if not calibrated.empty:
            x = calibrated["threshold"].iloc[0]
            y0 = calibrated["precision"].iloc[0]
            y1 = calibrated["f1"].iloc[0]
            axes[0].axvline(x, color=colors[model_name], alpha=0.25)
            axes[1].axvline(x, color=colors[model_name], alpha=0.25)
            axes[0].scatter([x], [y0], color=colors[model_name], s=50)
            axes[1].scatter([x], [y1], color=colors[model_name], s=50)

    axes[0].set_ylabel("Precision / Recall")
    axes[1].set_ylabel("F1")
    axes[1].set_xlabel("Anomali eşik değeri")
    axes[0].legend(loc="lower left", ncol=2)
    axes[1].legend(loc="lower left")
    _style_axes(axes[0])
    _style_axes(axes[1])
    _save_fig(fig, "05_kaist_rotating_threshold_sweep.png")


def _figure_kaist_rtf_trend(ctx: PackageContext) -> None:
    df = ctx.kaist_rtf_trend.copy()
    summary = ctx.kaist_rtf_threshold.copy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("KAIST Run-to-Failure: Anomali Trend ve Erken Uyarı", fontsize=18, weight="bold")
    pretty_names = {"isolation_forest": "Isolation Forest", "one_class_svm": "One-Class SVM"}
    colors = {"isolation_forest": "#2563EB", "one_class_svm": "#F97316"}

    for ax, model_name in zip(axes, ["isolation_forest", "one_class_svm"]):
        subset = df[df["model_name"] == model_name].sort_values("elapsed_hours")
        row = summary[summary["model_name"] == model_name].iloc[0]
        ax.plot(subset["elapsed_hours"], subset["rolling_anomaly_score_mean"], color=colors[model_name], lw=2.2)
        ax.axhline(row["calibrated_threshold"], color="#DC2626", linestyle="--", linewidth=1.7, label="Kalibre eşik")
        ax.axvspan(row["calibration_start_hour"], row["calibration_end_hour"], color="#DCFCE7", alpha=0.8, label="Kalibrasyon bölgesi")
        if not pd.isna(row["first_alarm_hour"]):
            ax.scatter([row["first_alarm_hour"]], [row["calibrated_threshold"]], color="#7C3AED", s=70, zorder=5, label="İlk alarm")
        ax.set_title(pretty_names[model_name])
        ax.set_ylabel("Yuvarlatılmış anomaly skoru")
        _style_axes(ax)
        ax.legend(loc="lower right")

    axes[1].set_xlabel("Geçen süre (saat)")
    _save_fig(fig, "06_kaist_rtf_anomali_trendi.png")


def _figure_nasa_ims_trend(ctx: PackageContext) -> None:
    trend = ctx.nasa_trend.copy()
    summary = ctx.nasa_threshold.copy()
    selected_sessions = [
        "nasa_ims_1st_test_bearing_3",
        "nasa_ims_1st_test_bearing_4",
        "nasa_ims_2nd_test_bearing_1",
        "nasa_ims_1st_test_bearing_1",
    ]
    session_labels = {
        "nasa_ims_1st_test_bearing_3": "1st_test / bearing 3 (dokümante inner race)",
        "nasa_ims_1st_test_bearing_4": "1st_test / bearing 4 (dokümante roller)",
        "nasa_ims_2nd_test_bearing_1": "2nd_test / bearing 1 (dokümante outer race)",
        "nasa_ims_1st_test_bearing_1": "1st_test / bearing 1 (referans örnek)",
    }
    palette = {
        "nasa_ims_1st_test_bearing_3": "#DC2626",
        "nasa_ims_1st_test_bearing_4": "#F97316",
        "nasa_ims_2nd_test_bearing_1": "#2563EB",
        "nasa_ims_1st_test_bearing_1": "#16A34A",
    }

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle("NASA IMS: Seçili Bearing Trajektorilerinde Anomali Trendleri", fontsize=18, weight="bold")
    pretty_names = {"isolation_forest": "Isolation Forest", "one_class_svm": "One-Class SVM"}

    for ax, model_name in zip(axes, ["isolation_forest", "one_class_svm"]):
        model_df = trend[(trend["model_name"] == model_name) & (trend["session_id"].isin(selected_sessions))]
        model_summary = summary[(summary["model_name"] == model_name) & (summary["session_id"].isin(selected_sessions))]
        threshold_value = float(model_summary["calibrated_threshold"].iloc[0])
        calib_end = float(model_summary["calibration_end_hour"].iloc[0])
        ax.axhline(threshold_value, color="#DC2626", linestyle="--", linewidth=1.6, label="Kalibre eşik")
        ax.axvspan(0, calib_end, color="#DCFCE7", alpha=0.8, label="Kalibrasyon bölgesi")
        for session_id in selected_sessions:
            subset = model_df[model_df["session_id"] == session_id].sort_values("elapsed_hours")
            if subset.empty:
                continue
            ax.plot(subset["elapsed_hours"], subset["rolling_anomaly_score_mean"], color=palette[session_id], lw=2, label=session_labels[session_id])
            alarm_row = model_summary[model_summary["session_id"] == session_id]
            if not alarm_row.empty and pd.notna(alarm_row["first_alarm_hour"].iloc[0]):
                x_alarm = float(alarm_row["first_alarm_hour"].iloc[0])
                alarm_point = subset.iloc[(subset["elapsed_hours"] - x_alarm).abs().argmin()]
                ax.scatter([alarm_point["elapsed_hours"]], [alarm_point["rolling_anomaly_score_mean"]], color=palette[session_id], edgecolor="black", s=55, zorder=6)
        ax.set_title(pretty_names[model_name])
        ax.set_ylabel("Yuvarlatılmış anomaly skoru")
        _style_axes(ax)
        ax.legend(loc="upper left", fontsize=9)

    axes[1].set_xlabel("Geçen süre (saat)")
    _save_fig(fig, "07_nasa_ims_anomali_trendi.png")


def _figure_ablation(ctx: PackageContext) -> None:
    df = ctx.smoke_ablation.copy().set_index("setup_name").loc[["ae_only", "vibration_only", "thermal_only", "fused"]]
    x = np.arange(len(df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title("Smoke-test Ablation: Tek Sensörler ve Fusion", fontsize=18, weight="bold")
    bars1 = ax.bar(x - width / 2, df["classifier_f1"], width, label="Classifier F1", color="#2563EB")
    bars2 = ax.bar(x + width / 2, df["anomaly_f1"], width, label="Anomaly F1", color="#F97316")
    ax.set_xticks(x, ["AE", "Titreşim", "Termal", "Fusion"])
    ax.set_ylim(0, 1.1)
    ax.legend()
    _style_axes(ax)
    _annotate_bars(ax, bars1)
    _annotate_bars(ax, bars2)
    _save_fig(fig, "08_smoke_test_ablation_ve_fusion_karsilastirmasi.png")


def _figure_feature_sensor_contribution(ctx: PackageContext) -> None:
    sensor = ctx.smoke_sensor_importance.copy().sort_values("normalized_importance", ascending=False)
    top = ctx.smoke_top_features.copy().sort_values("importance", ascending=True).tail(8)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Fusion Baseline: Sensör Katkısı ve Öne Çıkan Özellikler", fontsize=18, weight="bold")
    bars = axes[0].bar(sensor["sensor_group"], sensor["normalized_importance"], color=["#2563EB", "#F97316", "#10B981"])
    axes[0].set_title("Sensör grubu katkısı")
    axes[0].set_ylabel("Normalize önem")
    axes[0].set_ylim(0, 0.7)
    _style_axes(axes[0])
    _annotate_bars(axes[0], bars)
    axes[1].barh(top["feature"], top["importance"], color="#F97316")
    axes[1].set_title("En önemli fused özellikler")
    axes[1].set_xlabel("Önem")
    axes[1].tick_params(axis="y", labelsize=10)
    _style_axes(axes[1])
    _save_fig(fig, "09_feature_importance_ve_sensor_katkisi.png")


def _figure_framework_schema() -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    ax.set_title("Çoklu Veri Seti Araştırma Çerçevesi", fontsize=20, weight="bold")
    _draw_box(ax, (0.04, 0.68), 0.18, 0.15, "Smoke-test\ndemo pipeline", "#DBEAFE")
    _draw_box(ax, (0.04, 0.49), 0.18, 0.15, "KAIST\nRotating Machine", "#DBEAFE")
    _draw_box(ax, (0.04, 0.30), 0.18, 0.15, "KAIST\nRun-to-Failure", "#DBEAFE")
    _draw_box(ax, (0.04, 0.11), 0.18, 0.15, "NASA IMS", "#DBEAFE")
    _draw_box(ax, (0.28, 0.45), 0.2, 0.22, "Dataset registry\nAdapter katmanı\nModality bilgisi\nSemantik koruma", "#E0F2FE")
    _draw_box(ax, (0.54, 0.45), 0.2, 0.22, "Ortak deney şeması\nsession_id / group_id\nreference_region_role\nsession-safe split", "#DCFCE7")
    _draw_box(ax, (0.80, 0.56), 0.16, 0.14, "Supervised baseline\nuygun veri setlerinde", "#FCE7F3")
    _draw_box(ax, (0.80, 0.33), 0.16, 0.14, "Anomaly-first baseline\nIsolation Forest + OCSVM", "#F3E8FF")
    _draw_box(ax, (0.80, 0.10), 0.16, 0.14, "Raporlama ve yayın\nodaklı özetler", "#FEE2E2")

    for y in [0.755, 0.565, 0.375, 0.185]:
        _arrow(ax, (0.22, y), (0.28, 0.56))
    _arrow(ax, (0.48, 0.56), (0.54, 0.56))
    _arrow(ax, (0.74, 0.60), (0.80, 0.63))
    _arrow(ax, (0.74, 0.56), (0.80, 0.40))
    _arrow(ax, (0.74, 0.52), (0.80, 0.17))
    ax.text(0.5, 0.02, "Çerçeve, yeni veri seti eklemeyi kolaylaştırırken mevcut çalışan pipeline'ları korur.", ha="center", fontsize=12)
    _save_fig(fig, "10_coklu_veri_seti_arastirma_cercevesi.png")


def create_figures(ctx: PackageContext) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    _figure_system_architecture()
    _figure_smoke_test_summary(ctx)
    _figure_kaist_classifier_failure(ctx)
    _figure_kaist_anomaly_comparison(ctx)
    _figure_kaist_threshold_sweep(ctx)
    _figure_kaist_rtf_trend(ctx)
    _figure_nasa_ims_trend(ctx)
    _figure_ablation(ctx)
    _figure_feature_sensor_contribution(ctx)
    _figure_framework_schema()


def build_manifest() -> None:
    package_manifest = {
        "package_dir": str(PACKAGE_DIR.relative_to(ROOT)),
        "figure_count": len(list(FIGURES_DIR.glob("*.png"))),
        "table_csv_count": len(list(TABLES_DIR.glob("*.csv"))),
        "table_md_count": len(list(TABLES_DIR.glob("*.md"))),
        "selected_datasets": [
            "smoke-test demo",
            "KAIST rotating machine",
            "KAIST run-to-failure",
            "NASA IMS",
        ],
    }
    _save_json(package_manifest, PACKAGE_DIR / "package_manifest.json")


def main() -> None:
    _ensure_dirs()
    ctx = _read_context()
    build_tables(ctx)
    create_figures(ctx)
    build_manifest()


if __name__ == "__main__":
    main()
