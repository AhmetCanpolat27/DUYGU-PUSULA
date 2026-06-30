from pathlib import Path

input_path = Path("data/manual/user_examples.txt")
output_path = Path("data/manual/user_examples_ready.txt")

valid_labels = {"mutluluk", "uzuntu", "ofke", "korku", "saskinlik"}

def fix_encoding(text):
    if any(token in text for token in ["Ä", "Ã", "Å", "Â"]):
        for encoding in ["latin1", "cp1252"]:
            try:
                return text.encode(encoding).decode("utf-8")
            except Exception:
                pass
    return text

source = input_path.read_text(encoding="utf-8", errors="ignore")

clean_lines = []

for line in source.splitlines():
    line = line.strip()

    if not line:
        continue

    line = fix_encoding(line).strip()

    if line in valid_labels:
        continue

    if "|" not in line:
        continue

    text, label = line.rsplit("|", 1)
    text = text.strip()
    label = label.strip()

    if label not in valid_labels:
        continue

    if len(text) < 5:
        continue

    clean_lines.append(f"{text} | {label}")

output_path.write_text("\n".join(clean_lines) + "\n", encoding="utf-8")

print("Hazır dosya oluşturuldu:")
print(output_path)
print("Satır sayısı:", len(clean_lines))
print()
print("İlk 5 satır:")
for line in clean_lines[:5]:
    print(line)
