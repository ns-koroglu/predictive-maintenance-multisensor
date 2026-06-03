# Bulgular ve Tartışma Taslağı

## 1. Deneysel Değerlendirme Yaklaşımı

Bu tezde deneysel değerlendirme yalnızca raw accuracy değerine dayandırılmamıştır. Bunun temel nedeni, kestirimci bakım veri setlerinde sınıf dağılımının çoğu zaman dengeli olmaması ve çoğunluk sınıfını tahmin eden bir modelin yüzeyde yüksek accuracy üretebilmesidir. KAIST rotating machine deneyinde görüldüğü gibi, modelin tüm test örneklerini `faulty` tahmin etmesi yüksek raw accuracy oluşturabilmekte, ancak healthy sınıfı için hiçbir genelleme sağlamamaktadır.

Bu nedenle değerlendirmede balanced accuracy, macro F1, confusion matrix, threshold davranışı ve session/group-safe split sonuçları birlikte yorumlanmıştır. Session-safe veya group-safe split kullanımı, aynı oturumdan veya aynı bearing code'dan gelen örneklerin hem eğitim hem test kümesine düşmesini engelleyerek veri kaçağı riskini azaltır. Bu yaklaşım, özellikle pencereleme yapılan zaman serisi verilerinde kritik önemdedir.

Çalışmanın ana modelleme çizgisi feature-level fusion ve klasik makine öğrenmesi baselinelarıdır. Smoke-test düzeyinde AE, vibration ve thermal özniteliklerinin birlikte kullanımı pipeline'ın fusion kapasitesini göstermektedir. Public veri setlerinde ise veri setinin semantiğine göre daha konservatif davranılmıştır: label yapısı temiz olan Paderborn için supervised Random Forest baseline kurulmuş, dense label bulunmayan KAIST run-to-failure ve NASA IMS için anomaly-first değerlendirme kullanılmıştır.

Anomaly-first yaklaşım, özellikle az sayıda sağlıklı örnek bulunan veya arıza başlangıç zamanı açıkça etiketlenmemiş veri setlerinde daha savunulabilir bir çerçeve sunmaktadır. Bu yaklaşımda amaç doğrudan kesin arıza zamanı tahmin etmek değil, erken sağlıklı referans bölgesine göre davranış sapmasını ve threshold crossing örüntülerini raporlamaktır.

## 2. Veri Setlerinin Tezdeki Rolleri

### 2.1 Smoke-test Dataset

Smoke-test veri seti gerçek bir benchmark olarak değerlendirilmemelidir. Bu veri, pipeline'ın uçtan uca çalıştığını göstermek için kullanılmıştır. `results/raw_smoke_real_data_readiness/demo_summary.md` çıktısında session split, sınıflandırma, anomaly detection, fusion ablation ve feature importance raporlarının üretilebildiği görülmektedir. Bu nedenle smoke-test sonucu, gerçek public dataset performansı olarak değil, yazılım altyapısı ve deney akışının doğrulaması olarak sunulmalıdır.

### 2.2 Paderborn Dataset

Paderborn veri seti, mevcut proje içinde ilk temiz supervised public benchmark rolünü üstlenmektedir. Bu benchmark binary `healthy` vs `faulty` sınıflandırma olarak kurulmuştur. İlk baseline özellikle vibration-only tutulmuştur; current, thermal ve process kanalları veri setinde korunmakla birlikte bu ilk supervised deneyde kullanılmamıştır.

Değerlendirme group-aware split ile yapılmıştır ve `split_group = bearing_code` kullanılmıştır. Bu tercih, aynı bearing code'a ait ölçümlerin hem eğitim hem test kümesine düşmesini engeller. Bu nedenle Paderborn sonucu, leakage-aware supervised baseline olarak tezde kullanılabilir. Ancak performans düzeyi orta seviyededir ve nihai endüstriyel sınıflandırıcı iddiası taşımamalıdır.

### 2.3 KAIST Rotating Machine Dataset

KAIST rotating machine veri seti, sınıf dengesizliği altında supervised classifier sonucunun nasıl yanıltıcı olabileceğini gösteren kritik bir örnektir. İlk baseline vibration + thermal öznitelikleriyle kurulmuştur. Current verisi korunmuş, ancak ilk baseline'da kullanılmamıştır. Acoustic verinin AE olarak adlandırılmaması önemlidir; KAIST acoustic branch, acoustic veridir ve acoustic emission yerine geçecek şekilde yorumlanmamalıdır.

Bu veri setinde session'lar condition-matched olarak ele alınmıştır; ancak modality'lerin güvenli biçimde time-synchronized olduğu iddia edilmemiştir. Bu nedenle sonuçlar condition-matched fakat unsynchronized çoklu sensör verisi olarak yorumlanmalıdır. Supervised classifier'ın healthy sınıfında başarısız olması, anomaly-first çerçevenin bu veri seti için daha uygun olduğunu desteklemektedir.

### 2.4 KAIST Run-to-Failure Dataset

KAIST run-to-failure veri seti dense supervised classification veri seti olarak ele alınmamıştır. Veri, kronolojik bir degradation/anomaly sequence olarak değerlendirilmiştir. Deneyde vibration x/y, bearing temperature ve ambient temperature öznitelikleri kullanılmıştır.

Bu veri setinde ilk 24 saatlik bölüm healthy-reference calibration bölgesi olarak kullanılmıştır. Daha sonraki saatlerde gözlenen threshold crossing davranışı, erken referans rejiminden sapma olarak yorumlanmalıdır. Bu sonuçlar doğrulanmış arıza başlangıç zamanı veya kesin RUL tahmini olarak sunulmamalıdır.

### 2.5 NASA IMS Dataset

NASA IMS veri seti bearing-level anomaly progression analizi için kullanılmıştır. Her bearing trajectory ayrı session olarak, test run ise group semantiğiyle ele alınmıştır. Snapshot-level dense label bulunmadığı için supervised classification uygulanmamıştır.

Documented end-of-run failure bilgileri metadata olarak korunmuştur; ancak bu bilgiler her snapshot için dense target label gibi kullanılmamıştır. Bu nedenle NASA IMS sonuçları, erken healthy-reference bölgesine göre sapma ve alarm karakteristiği analizi olarak değerlendirilmelidir.

## 3. Paderborn Supervised Benchmark Bulguları

Paderborn supervised benchmark sonucu, binary healthy/faulty sınıflandırma için orta düzeyde bir baseline sunmaktadır. Deney vibration-only Random Forest ile yürütülmüş ve group-aware split kullanılmıştır. Train tarafında 22 group ve 1759 session, test tarafında 8 group ve 640 session vardır. Group overlap bulunmamaktadır.

Processed veri seti dağılımı `faulty`: 1919 ve `healthy`: 480 şeklindedir. Deney splitinde train dağılımı `faulty`: 1439, `healthy`: 320; test dağılımı ise `faulty`: 480, `healthy`: 160 olarak raporlanmıştır. Bu dağılım, raw accuracy'nin tek başına yeterli olmadığını göstermektedir.

| Metrik | Değer | Yorum |
|---|---:|---|
| Accuracy | 0.7031 | Çoğunluk sınıfı etkisi nedeniyle tek başına başarı göstergesi olarak kullanılmamalıdır. |
| Balanced accuracy | 0.5917 | İki sınıfı daha dengeli değerlendirdiği için daha dürüst ana metriklerden biridir. |
| Macro F1 | 0.5938 | Sınıflar arası performans farkını daha açık gösterir; orta düzey baseline performansına işaret eder. |
| Macro precision | 0.5967 | Genel ayırt etme kapasitesi sınırlı fakat tamamen rastgele değildir. |
| Macro recall | 0.5917 | Healthy ve faulty sınıflarında dengeli genellemenin hâlâ zor olduğunu gösterir. |
| ROC-AUC | 0.8358 | Probability ranking tarafında bilgi olduğunu gösterir, ancak confusion matrix ile birlikte yorumlanmalıdır. |
| PR-AUC | 0.9509 | Pozitif sınıf `faulty` için yüksek görünür; sınıf dağılımı nedeniyle dikkatli yorumlanmalıdır. |

Confusion matrix, label sırası `faulty`, `healthy` olacak şekilde şu şekildedir:

| True \ Predicted | faulty | healthy |
|---|---:|---:|
| faulty | 391 | 89 |
| healthy | 101 | 59 |

Bu matrise göre model faulty örneklerin önemli bir kısmını doğru yakalamaktadır; ancak healthy sınıfında belirgin zayıflık vardır. Healthy sınıfı için F1 değeri 0.3831 olarak raporlanmıştır. Bu durum, modelin özellikle sağlıklı durumları ayırt etmede zorlandığını göstermektedir.

Per-group değerlendirme de bu yorumu desteklemektedir. Test gruplarında `paderborn_K001` ve `paderborn_K006` healthy olmasına rağmen majority prediction `faulty` çıkmıştır; bu false positive riskini gösterir. Buna karşılık `paderborn_KI05` faulty olduğu halde majority prediction `healthy` çıkmıştır; bu da false negative riskinin hâlâ bulunduğunu gösterir.

Sonuç olarak Paderborn, tezde temiz ve leakage-aware supervised baseline olarak kullanılabilir. Ancak bu sonuç final endüstriyel classifier başarısı olarak değil, group-aware split altında orta düzey ve açıklanabilir bir başlangıç benchmarkı olarak sunulmalıdır.

## 4. KAIST Rotating Machine Bulguları

KAIST rotating machine deneyinde supervised classifier sonucu, raw accuracy'nin ağır sınıf dengesizliği altında yanıltıcı olabileceğini açık biçimde göstermektedir. Deneyde session-safe split kullanılmış, train/test arasında session overlap olmadığı raporlanmıştır. Buna rağmen test seti dengesizdir: test pencerelerinin 940'ı `faulty`, 118'i `healthy` sınıfındadır.

Random Forest classifier için raporlanan raw accuracy 0.8885'tir. İlk bakışta bu değer yüksek görünmektedir; ancak confusion matrix modelin tüm test örneklerini `faulty` tahmin ettiğini göstermektedir. Bu nedenle healthy recall sıfırdır. Balanced accuracy 0.5000 ve macro F1 0.4705 değerleri, modelin aslında sağlıklı sınıfa genelleyemediğini daha doğru biçimde ortaya koymaktadır.

| Metric | Classifier result | Interpretation |
|---|---:|---|
| Accuracy | 0.8885 | Test seti çoğunlukla `faulty` olduğu için yanıltıcı biçimde yüksek görünmektedir. |
| Balanced accuracy | 0.5000 | İki sınıf dengeli değerlendirildiğinde modelin ayırt etme kapasitesi zayıftır. |
| Macro precision | 0.4442 | Healthy sınıfında tahmin üretilemediği için makro düzeyde düşüktür. |
| Macro recall | 0.5000 | Healthy recall sıfır olduğu için sınıf dengesi açısından başarısızdır. |
| Macro F1 | 0.4705 | Supervised classifier'ın genel dengeli performansının yetersiz olduğunu gösterir. |
| ROC-AUC | 0.3956 | Sınıf ayrımı açısından zayıf bir ranking davranışına işaret eder. |
| PR-AUC | 0.8655 | Pozitif sınıf dağılımı nedeniyle tek başına güçlü başarı iddiası için yeterli değildir. |

Confusion matrix:

| True \ Predicted | faulty | healthy |
|---|---:|---:|
| faulty | 940 | 0 |
| healthy | 118 | 0 |

Bu bulgunun temel sonucu şudur: ağır sınıf dengesizliği altında raw accuracy tek başına güvenilir değildir. Modelin tüm test örneklerini çoğunluk sınıfına ataması, yüksek accuracy üretmesine rağmen bakım açısından kritik olan healthy/fault ayrımını sağlayamamaktadır.

Bu nedenle KAIST rotating machine için anomaly-first değerlendirme daha savunulabilir bir çerçeve sunar. Isolation Forest, default reported anomaly baseline olarak seçilmiştir. Calibrated threshold 0.6131, balanced accuracy 0.5268, precision 0.8968, recall 0.6468 ve F1 0.7515 olarak raporlanmıştır. Bu sonuç supervised classifier'a göre daha anlamlı bir alarm davranışı sunsa da yine konservatif yorumlanmalıdır.

Threshold sweep sonucunda Isolation Forest için önerilen threshold 0.6230 ve balanced accuracy 0.7453 olarak raporlanmıştır. Ancak bu değer held-out sweep üzerinden seçildiği için optimistic analysis niteliğindedir. Tezde train-only calibrated operating point ile sweep-selected threshold ayrı tutulmalıdır.

## 5. KAIST RTF ve NASA IMS Anomaly-First Bulguları

KAIST run-to-failure ve NASA IMS deneyleri supervised classification yerine anomaly-first progression analizi olarak değerlendirilmiştir. Bu tercih, veri setlerinde snapshot veya hour seviyesinde dense label bulunmaması nedeniyle gereklidir. Label uydurmak yerine erken sağlıklı referans bölgesi seçilmiş, anomaly modelleri bu referans bölgesine göre kalibre edilmiş ve daha sonraki zaman adımlarında sapma davranışı incelenmiştir.

KAIST run-to-failure deneyinde 129 saatlik feature row üretilmiş ve ilk 24 saat calibration bölgesi olarak kullanılmıştır. Isolation Forest için calibrated threshold 0.711477 olarak raporlanmış ve alarm oluşmamıştır. One-Class SVM için threshold 0.022606, first threshold crossing hour 24.0 ve first sustained warning hour 26.0 olarak raporlanmıştır. Bu bulgu, One-Class SVM'in daha erken ve daha sürekli alarm ürettiğini göstermektedir; ancak bu alarm doğrulanmış failure prediction olarak yorumlanmamalıdır.

NASA IMS deneyinde 3 run, 12 bearing session ve 37856 feature row raporlanmıştır. Vibration-only anomaly progression analizi yapılmıştır. Calibration bölgesinde toplam 1780 reference file kullanılmıştır. Isolation Forest için threshold 0.639746 ve sessions_with_alarm 4; One-Class SVM için threshold 2.244755 ve sessions_with_alarm 12 olarak raporlanmıştır. Bu fark, modellerin alarm karakteristiklerinin farklı olduğunu göstermektedir. Isolation Forest daha konservatif alarm üretirken, One-Class SVM daha geniş alarm davranışı göstermiştir.

Bu sonuçlar erken uyarı araştırması açısından yararlıdır çünkü sistemin normal referans rejiminden ne zaman ve nasıl saptığını gösterir. Ancak veri setlerinde dense failure-onset label bulunmadığı için bu sapmalar kesin arıza başlangıcı veya RUL tahmini olarak sunulmamalıdır. Doğru yorum, “erken healthy-reference bölgesine göre anomali eğilimi ve threshold crossing davranışı” şeklindedir.

## 6. Genel Tartışma

Session/group-safe split bu çalışmanın en önemli metodolojik bileşenlerinden biridir. Zaman serisi verilerinde aynı oturumdan üretilen pencereler birbirine yüksek derecede benzer olabilir. Bu pencerelerin hem eğitim hem test kümesine girmesi model performansını yapay olarak artırabilir. Bu nedenle KAIST rotating machine için session-safe split, Paderborn için bearing-code group-aware split kullanılması tez açısından kritik bir güvenilirlik önlemidir.

Fabricated label üretimi, özellikle run-to-failure ve progression veri setlerinde ciddi bir metodolojik risktir. KAIST RTF ve NASA IMS gibi veri setlerinde her snapshot veya hour için kesin healthy/faulty etiketi bulunmadığında, bu etiketleri sonradan uydurmak model performansını bilimsel olarak tartışmalı hale getirir. Bu tezde bu riskten kaçınılmış ve dense label olmayan veri setleri anomaly-first çerçevede yorumlanmıştır.

Threshold sensitivity de anomaly-first deneylerde önemli bir tartışma başlığıdır. Aynı anomaly score dizisi, seçilen threshold'a göre farklı alarm davranışları üretebilir. KAIST rotating machine deneyinde calibrated threshold ile sweep-selected threshold arasındaki fark bu durumu göstermektedir. Bu nedenle threshold seçiminin hangi veri üzerinde ve hangi amaçla yapıldığı açıkça raporlanmalıdır.

Klasik makine öğrenmesi modellerinin kullanılması bu tez için uygundur. Random Forest, Isolation Forest ve One-Class SVM; küçük veri koşullarında, açıklanabilirlik gereksiniminde ve tez ölçeğinde savunulabilir baselinelardır. Bu çalışma yeni bir deep learning modeli önermekten ziyade, veri seti semantiğine uygun, leakage-aware ve class imbalance-aware bir değerlendirme altyapısı kurmaya odaklanmaktadır.

Genel olarak anomaly-first framing, classification-first framing'e göre daha güçlüdür; çünkü mevcut public veri setlerinin bir kısmında dense supervised label bulunmamaktadır ve bazı veri setlerinde sınıf dengesizliği supervised classifier sonucunu yanıltıcı hale getirmektedir. Bu nedenle tezde supervised classification yalnızca Paderborn gibi label yapısı daha temiz veri setlerinde benchmark olarak kullanılmalı, progression veri setleri ise anomaly trend ve early warning bağlamında tartışılmalıdır.

## 7. Tez Katkısı

1. Çoklu veri setleri için session/group-safe ve leakage-aware değerlendirme çerçevesi kurulmuştur.
2. Ağır sınıf dengesizliği altında raw accuracy'nin yanıltıcı olabileceği KAIST rotating machine üzerinden deneysel olarak gösterilmiştir.
3. Dense label bulunmayan progression veri setleri için anomaly-first yorumlama hattı uygulanmıştır.
4. Paderborn üzerinde binary healthy/faulty, vibration-only ve group-aware supervised baseline oluşturulmuştur.
5. Feature-level fusion ve Random Forest feature importance tabanlı açıklanabilirlik, tez ölçeğinde sade ve savunulabilir bir baseline olarak entegre edilmiştir.

## 8. Sınırlılıklar

- Henüz gerçek özel laboratuvar verisi toplanmamıştır.
- AE donanımıyla tam deneysel validasyon yapılmamıştır.
- KAIST acoustic verisi AE değildir ve AE sonucu gibi sunulmamalıdır.
- KAIST RTF ve NASA IMS gibi veri setlerinde snapshot/hour seviyesinde dense label yoktur.
- Çalışma endüstriyel deployment veya gerçek zamanlı saha sistemi iddiası taşımamaktadır.
- Nihai RUL tahmini yapılmamıştır.
- Model performansı bazı veri setlerinde orta veya zayıf düzeydedir; özellikle KAIST rotating supervised classifier sonucu başarısızlık örneği olarak yorumlanmalıdır.
- Threshold seçimi anomaly-first sonuçları belirgin biçimde etkileyebilmektedir.

## 9. Sonraki Çalışmalar

1. Gerçek laboratuvar veri toplama düzeneği kurulmalı ve kontrollü arıza senaryoları kaydedilmelidir.
2. AE, vibration ve thermal sensörleri senkronize veya en azından zaman damgası güvenilir biçimde eşleştirilebilir şekilde toplanmalıdır.
3. Anomaly threshold calibration stratejileri daha sistematik karşılaştırılmalıdır.
4. SHAP veya daha derin açıklanabilirlik yöntemleri, mevcut sade RF importance çıktıları korunarak opsiyonel olarak değerlendirilebilir.
5. XGBoost veya Autoencoder karşılaştırmaları yalnızca scope uygunsa ve mevcut classical baseline sonuçlarını gölgelemeyecek şekilde eklenmelidir.
6. CWRU adapter yalnızca kapsam izin verirse eklenmeli; parser yazmadan önce yerel veri yapısı için ayrı specification hazırlanmalıdır.

## 10. Hocaya Söylenecek Kısa Teknik Özet

Bu aşamada proje, sadece çalışan bir demo değil, çoklu veri setlerini destekleyen ve veri kaçağına dikkat eden bir kestirimci bakım değerlendirme altyapısı haline geldi. Değerlendirmeyi özellikle raw accuracy üzerine kurmadım; çünkü KAIST rotating machine sonucunda bunun ne kadar yanıltıcı olabildiği açıkça görülüyor. Orada classifier 0.8885 accuracy veriyor, fakat confusion matrix tüm test örneklerinin `faulty` tahmin edildiğini gösteriyor. Bu yüzden balanced accuracy 0.5000 ve macro F1 0.4705 daha doğru yorum sağlıyor. Paderborn tarafında ise ilk temiz supervised benchmark mevcut: binary healthy/faulty, vibration-only ve bearing-code group-aware split ile Random Forest baseline kuruldu. Burada accuracy 0.7031, balanced accuracy 0.5917 ve macro F1 0.5938; yani orta düzey ama leakage-safe bir başlangıç sonucu var. KAIST run-to-failure ve NASA IMS tarafında dense label uydurmadım; bunları supervised classification gibi değil, erken healthy-reference bölgesine göre anomaly trend ve threshold crossing analizi olarak ele aldım. Bu nedenle tezin ana katkısı yeni bir deep learning modeli değil; class imbalance, leakage riski ve label belirsizliği altında daha güvenli ve savunulabilir bir değerlendirme çerçevesi kurmak.
