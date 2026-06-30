# DuyguPusula

**DuyguPusula**, Türkçe metinlerde duygu analizi yapmak için geliştirilmiş BERTurk tabanlı bir uygulamadır. Girilen metnin baskın duygusunu tahmin eder ve tüm duygu sınıflarına ait güven skorlarını gösterir.

## Proje Bilgileri

**Proje Adı:** DuyguPusula
**Geliştiren:** Ahmet Canpolat
**Öğrenci No:** 222523024
**Model:** BERTurk
**Arayüz:** Streamlit
**Dil:** Python

## Desteklenen Duygu Sınıfları

* Mutluluk
* Üzüntü
* Öfke
* Korku
* Şaşkınlık

## Proje Yapısı

```text
AhmetCanpolat_222523024_DuyguPusula/
│
├── app.py
├── main_berturk.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── config.py
│   ├── predict_berturk.py
│   ├── train_berturk.py
│   └── ...
│
├── models/
│   └── berturk/
│
└── data/
    ├── processed/
    └── manual/
```

## Kurulum

```powershell
cd AhmetCanpolat_222523024_DuyguPusula
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Streamlit Arayüzünü Çalıştırma

```powershell
streamlit run app.py
```

## Konsol Üzerinden Çalıştırma

```powershell
python main_berturk.py
```

Programdan çıkmak için:

```text
q
```

## Model Klasörü

Eğitilmiş model proje içinde aşağıdaki klasörde bulunmaktadır:

```text
models/berturk
```

Model klasörü proje içinde bulunduğu için uygulamayı çalıştırmak için yeniden eğitim yapmak gerekli değildir.

## Yeniden Eğitim

```powershell
python -m src.train_berturk
```

Eğitim tamamlandığında model tekrar aşağıdaki klasöre kaydedilir:

```text
models/berturk
```

## Test

```powershell
python test_50.py
```

## Kullanılan Kütüphaneler

```text
pandas
numpy
scikit-learn
matplotlib
joblib
streamlit
transformers
datasets
torch
accelerate
tqdm
rich
safetensors
openpyxl
sentencepiece
protobuf
```

## Model Performansı

Son eğitim sonucunda elde edilen temel başarı değerleri:

```text
Test Accuracy: 0.9763
Test Macro F1: 0.9762
Test Weighted F1: 0.9762
```

Manuel 50 cümlelik test sonucunda:

```text
48 / 50 doğru tahmin
%96 başarı oranı
```

## Örnek Kullanım

Girdi:

```text
Birden tır korna çaldı, ödüm koptu.
```

Çıktı:

```text
Ana duygu: Korku
Güven: %99.90
```

## Notlar

* Python 3.11 kullanılması önerilir.
* Streamlit arayüzü `app.py` üzerinden çalıştırılır.
* Konsol uygulaması `main_berturk.py` üzerinden çalıştırılır.
* Eğitilmiş model `models/berturk` klasöründe yer alır.
* Detaylı açıklamalar proje raporunda yer alacaktır.
