import re
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DATA_PATH, TEXT_COLUMN, LABEL_COLUMN


READY_PATH = Path("data/manual/user_examples_ready.txt")
CLEAN_PATH = Path("data/manual/user_examples_clean.csv")
TRAIN_EXTRA_PATH = Path("data/manual/train_extra_v2.csv")
MANUAL_TEST_PATH = Path("data/manual/manual_test_v2.csv")
AUGMENTED_V2_PATH = Path("data/processed/turkish_tweet_emotion_augmented_v2.csv")

VALID_LABELS = {"mutluluk", "uzuntu", "ofke", "korku", "saskinlik"}

LABEL_ALIASES = {
    "mutluluk": "mutluluk",
    "mutlu": "mutluluk",
    "uzuntu": "uzuntu",
    "üzüntü": "uzuntu",
    "uzgun": "uzuntu",
    "üzgün": "uzuntu",
    "ofke": "ofke",
    "öfke": "ofke",
    "kizgin": "ofke",
    "kızgın": "ofke",
    "korku": "korku",
    "saskinlik": "saskinlik",
    "şaşkınlık": "saskinlik",
    "saskin": "saskinlik",
    "şaşkın": "saskinlik",
    "surpriz": "saskinlik",
    "sürpriz": "saskinlik"
}


def clean_value(value):
    value = str(value).strip()
    value = value.strip('"').strip("'").strip()
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_label(value):
    label = clean_value(value).lower()
    return LABEL_ALIASES.get(label, label)


def load_user_examples():
    rows = []
    rejected = []

    source = READY_PATH.read_text(encoding="utf-8", errors="ignore")

    for line_number, line in enumerate(source.splitlines(), start=1):
        line = clean_value(line)

        if not line:
            continue

        if "|" not in line:
            rejected.append((line_number, line, "ayrac_yok"))
            continue

        text, label = line.rsplit("|", 1)
        text = clean_value(text)
        label = normalize_label(label)

        if label not in VALID_LABELS:
            rejected.append((line_number, line, "gecersiz_etiket"))
            continue

        if len(text) < 12:
            rejected.append((line_number, line, "cok_kisa"))
            continue

        rows.append({
            TEXT_COLUMN: text,
            LABEL_COLUMN: label,
            "source": "manual_user",
            "line_number": line_number
        })

    frame = pd.DataFrame(rows)

    if frame.empty:
        raise ValueError("Geçerli kullanıcı verisi bulunamadı.")

    frame["text_key"] = frame[TEXT_COLUMN].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    frame = frame.drop_duplicates(subset=["text_key", LABEL_COLUMN])
    frame = frame.drop(columns=["text_key"])
    frame = frame.sample(frac=1, random_state=42).reset_index(drop=True)

    rejected_frame = pd.DataFrame(rejected, columns=["line_number", "line", "reason"])

    return frame, rejected_frame


def split_user_examples(frame):
    train_frame, manual_test_frame = train_test_split(
        frame,
        test_size=0.20,
        random_state=42,
        stratify=frame[LABEL_COLUMN]
    )

    train_frame = train_frame.reset_index(drop=True)
    manual_test_frame = manual_test_frame.reset_index(drop=True)

    return train_frame, manual_test_frame


def build_augmented_dataset(train_frame):
    base_frame = pd.read_csv(PROCESSED_DATA_PATH)

    if "raw_label" in base_frame.columns:
        base_frame = base_frame[[TEXT_COLUMN, "raw_label", LABEL_COLUMN]].copy()
        extra_frame = train_frame[[TEXT_COLUMN, LABEL_COLUMN]].copy()
        extra_frame["raw_label"] = extra_frame[LABEL_COLUMN]
        extra_frame = extra_frame[[TEXT_COLUMN, "raw_label", LABEL_COLUMN]]
    else:
        base_frame = base_frame[[TEXT_COLUMN, LABEL_COLUMN]].copy()
        extra_frame = train_frame[[TEXT_COLUMN, LABEL_COLUMN]].copy()

    merged = pd.concat([base_frame, extra_frame], ignore_index=True)
    merged["text_key"] = merged[TEXT_COLUMN].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    merged = merged.drop_duplicates(subset=["text_key", LABEL_COLUMN])
    merged = merged.drop(columns=["text_key"])
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    return merged


def main():
    frame, rejected_frame = load_user_examples()
    train_frame, manual_test_frame = split_user_examples(frame)
    augmented_frame = build_augmented_dataset(train_frame)

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUGMENTED_V2_PATH.parent.mkdir(parents=True, exist_ok=True)

    frame.to_csv(CLEAN_PATH, index=False, encoding="utf-8-sig")
    train_frame.to_csv(TRAIN_EXTRA_PATH, index=False, encoding="utf-8-sig")
    manual_test_frame.to_csv(MANUAL_TEST_PATH, index=False, encoding="utf-8-sig")
    augmented_frame.to_csv(AUGMENTED_V2_PATH, index=False, encoding="utf-8-sig")

    if not rejected_frame.empty:
        rejected_frame.to_csv(Path("data/manual/rejected_user_examples.csv"), index=False, encoding="utf-8-sig")

    print("Kullanıcı verisi işlendi.")
    print(f"Temiz örnek sayısı: {len(frame)}")
    print(f"Eğitime ayrılan örnek sayısı: {len(train_frame)}")
    print(f"Manuel teste ayrılan örnek sayısı: {len(manual_test_frame)}")
    print(f"Genişletilmiş veri seti satır sayısı: {len(augmented_frame)}")
    print()
    print("Temiz veri dağılımı:")
    print(frame[LABEL_COLUMN].value_counts())
    print()
    print("Eğitim ek veri dağılımı:")
    print(train_frame[LABEL_COLUMN].value_counts())
    print()
    print("Manuel test dağılımı:")
    print(manual_test_frame[LABEL_COLUMN].value_counts())
    print()
    print("Augmented v2 toplam dağılımı:")
    print(augmented_frame[LABEL_COLUMN].value_counts())

    if not rejected_frame.empty:
        print()
        print(f"Reddedilen satır sayısı: {len(rejected_frame)}")


if __name__ == "__main__":
    main()
