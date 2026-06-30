from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MANUAL_DIR = ROOT / "data" / "manual"

MANUAL_CANDIDATES = [
    MANUAL_DIR / "user_examples",
    MANUAL_DIR / "user_examples.txt",
    MANUAL_DIR / "user_examples.csv",
]

MANUAL_CLEAN_PATH = MANUAL_DIR / "user_examples_clean.csv"
MANUAL_READY_PATH = MANUAL_DIR / "user_examples_ready.csv"

PROCESSED_PATH = ROOT / "data" / "processed" / "turkish_tweet_emotion_processed.csv"
AUGMENTED_PATH = ROOT / "data" / "processed" / "turkish_tweet_emotion_augmented.csv"

allowed_labels = {"mutluluk", "uzuntu", "ofke", "korku", "saskinlik"}

def norm_label(label):
    label = str(label).strip().lower()
    label = label.replace("ı", "i")
    label = label.replace("ğ", "g")
    label = label.replace("ü", "u")
    label = label.replace("ş", "s")
    label = label.replace("ö", "o")
    label = label.replace("ç", "c")
    return label

def clean_text(text):
    text = str(text).strip()
    text = " ".join(text.split())
    return text

def kolon_bul(df, adaylar):
    for kolon in adaylar:
        if kolon in df.columns:
            return kolon
    return None

def manual_dosya_bul():
    for path in MANUAL_CANDIDATES:
        if path.exists():
            return path

    bulunanlar = list(MANUAL_DIR.glob("user_examples*"))

    if bulunanlar:
        return bulunanlar[0]

    raise FileNotFoundError(f"user_examples dosyası bulunamadı. Klasör: {MANUAL_DIR}")

def manuel_veriyi_oku():
    manual_path = manual_dosya_bul()
    print(f"Okunan manuel dosya: {manual_path}")

    if manual_path.suffix.lower() == ".csv":
        raw_df = pd.read_csv(manual_path)

        text_col = kolon_bul(raw_df, ["text", "metin", "sentence", "tweet", "content", "clean_text"])
        label_col = kolon_bul(raw_df, ["label", "duygu", "emotion", "target", "class"])

        if text_col is None or label_col is None:
            raise ValueError(f"CSV içinde metin veya etiket kolonu bulunamadı. Kolonlar: {list(raw_df.columns)}")

        rows = []

        for _, row in raw_df.iterrows():
            text = clean_text(row[text_col])
            label = norm_label(row[label_col])

            if text and label in allowed_labels:
                rows.append({"text": text, "label": label})

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["text", "label"])
        return df

    lines = manual_path.read_text(encoding="utf-8-sig").splitlines()
    rows = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if "|" not in line:
            continue

        text, label = line.rsplit("|", 1)
        text = clean_text(text)
        label = norm_label(label)

        if not text:
            continue

        if label not in allowed_labels:
            continue

        rows.append({"text": text, "label": label})

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["text", "label"])
    return df

def label_id_hazirla(df, label_col):
    for possible_id_col in ["label_id", "target_id", "class_id"]:
        if possible_id_col in df.columns:
            id_map = {}

            for label in allowed_labels:
                secilenler = df[df[label_col].astype(str).apply(norm_label) == label][possible_id_col].dropna()

                if len(secilenler) > 0:
                    id_map[label] = secilenler.iloc[0]

            return possible_id_col, id_map

    return None, {}

def hedefe_ekle(target_path, manual_df):
    if not target_path.exists():
        print(f"Atlandı, dosya yok: {target_path}")
        return

    df = pd.read_csv(target_path)

    text_col = kolon_bul(df, ["text", "metin", "sentence", "tweet", "content", "clean_text"])
    label_col = kolon_bul(df, ["label", "duygu", "emotion", "target", "class"])

    if text_col is None or label_col is None:
        raise ValueError(f"{target_path.name} içinde metin veya etiket kolonu bulunamadı. Kolonlar: {list(df.columns)}")

    backup_path = target_path.with_name(target_path.stem + "_backup_before_manual_update.csv")

    if not backup_path.exists():
        shutil.copy(target_path, backup_path)

    id_col, id_map = label_id_hazirla(df, label_col)

    new_rows = []

    for _, item in manual_df.iterrows():
        row = {col: "" for col in df.columns}
        row[text_col] = item["text"]
        row[label_col] = item["label"]

        if id_col is not None and item["label"] in id_map:
            row[id_col] = id_map[item["label"]]

        new_rows.append(row)

    new_df = pd.DataFrame(new_rows)
    merged = pd.concat([df, new_df], ignore_index=True)

    merged[text_col] = merged[text_col].astype(str).apply(clean_text)
    merged[label_col] = merged[label_col].astype(str).apply(norm_label)

    merged = merged.drop_duplicates(subset=[text_col, label_col])
    merged.to_csv(target_path, index=False, encoding="utf-8-sig")

    print(f"\nGüncellendi: {target_path}")
    print(f"Yedek: {backup_path}")
    print(f"Toplam satır: {len(merged)}")
    print(merged[label_col].value_counts())

def main():
    manual_df = manuel_veriyi_oku()

    if len(manual_df) == 0:
        raise ValueError("Manuel dosyadan geçerli veri okunamadı.")

    MANUAL_CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)

    manual_df.to_csv(MANUAL_CLEAN_PATH, index=False, encoding="utf-8-sig")
    manual_df.to_csv(MANUAL_READY_PATH, index=False, encoding="utf-8-sig")

    print("\nManuel veri hazırlandı.")
    print(f"Toplam manuel örnek: {len(manual_df)}")
    print(manual_df["label"].value_counts())

    hedefe_ekle(PROCESSED_PATH, manual_df)

    if not AUGMENTED_PATH.exists() and PROCESSED_PATH.exists():
        shutil.copy(PROCESSED_PATH, AUGMENTED_PATH)

    hedefe_ekle(AUGMENTED_PATH, manual_df)

    print("\nManuel örnekler eğitim verisine eklendi.")

if __name__ == "__main__":
    main()
