import re
import urllib.request
import pandas as pd

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TEXT_COLUMN, LABEL_COLUMN, RAW_LABEL_COLUMN


DATASET_URL = "https://huggingface.co/datasets/anilguven/turkish_tweet_emotion_dataset/resolve/main/Turkish_Tweet_Dataset.csv"

TARGET_LABELS = {"ofke", "mutluluk", "uzuntu", "korku", "saskinlik"}


def decode_content(content):
    for encoding in ["utf-8-sig", "utf-8", "cp1254", "latin-1"]:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            pass
    return content.decode("utf-8", errors="ignore")


def normalize_label(label):
    value = str(label).strip().lower()
    value = value.replace('"', "").replace("'", "")
    value = value.translate(str.maketrans({
        "ı": "i",
        "ğ": "g",
        "ü": "u",
        "ş": "s",
        "ö": "o",
        "ç": "c"
    }))

    mapping = {
        "kizgin": "ofke",
        "kızgın": "ofke",
        "ofke": "ofke",
        "öfke": "ofke",
        "anger": "ofke",
        "angry": "ofke",

        "mutlu": "mutluluk",
        "mutluluk": "mutluluk",
        "sevinc": "mutluluk",
        "sevinç": "mutluluk",
        "happy": "mutluluk",
        "happiness": "mutluluk",

        "uzgun": "uzuntu",
        "üzgün": "uzuntu",
        "uzuntu": "uzuntu",
        "üzüntü": "uzuntu",
        "sad": "uzuntu",
        "sadness": "uzuntu",

        "korku": "korku",
        "fear": "korku",
        "scared": "korku",

        "saskin": "saskinlik",
        "şaşkın": "saskinlik",
        "saskinlik": "saskinlik",
        "şaşkınlık": "saskinlik",
        "sasirma": "saskinlik",
        "şaşırma": "saskinlik",
        "surprise": "saskinlik",
        "surprised": "saskinlik",
        "surpriz": "saskinlik",
        "sürpriz": "saskinlik"
    }

    return mapping.get(value, value)


def clean_text(text):
    value = str(text).strip()
    value = value.replace("\ufeff", "")
    value = value.strip('"').strip("'").strip()
    value = re.sub(r"\s+", " ", value)
    return value


def parse_lines(source_text):
    rows = []

    for line in source_text.splitlines():
        line = line.strip()

        if not line:
            continue

        if "," not in line:
            continue

        text, raw_label = line.rsplit(",", 1)

        text = clean_text(text)
        raw_label = clean_text(raw_label)
        label = normalize_label(raw_label)

        if text.lower() in {"text", "tweet", "metin"}:
            continue

        if label not in TARGET_LABELS:
            continue

        if len(text) < 3:
            continue

        rows.append({
            TEXT_COLUMN: text,
            RAW_LABEL_COLUMN: raw_label,
            LABEL_COLUMN: label
        })

    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=[TEXT_COLUMN, LABEL_COLUMN])
    frame = frame.reset_index(drop=True)
    return frame


def prepare_dataset():
    with urllib.request.urlopen(DATASET_URL) as response:
        content = response.read()

    source_text = decode_content(content)
    RAW_DATA_PATH.write_text(source_text, encoding="utf-8-sig")

    processed_frame = parse_lines(source_text)
    processed_frame.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8-sig")

    print("Veri seti indirildi ve hazırlandı.")
    print(f"Ham veri: {RAW_DATA_PATH}")
    print(f"İşlenmiş veri: {PROCESSED_DATA_PATH}")
    print(f"Satır sayısı: {len(processed_frame)}")
    print()
    print("Duygu dağılımı:")
    print(processed_frame[LABEL_COLUMN].value_counts())
    print()
    print("Ham etiketler:")
    print(processed_frame[RAW_LABEL_COLUMN].value_counts())


if __name__ == "__main__":
    prepare_dataset()
