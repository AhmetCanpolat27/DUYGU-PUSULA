import json
from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from src.config import TEXT_COLUMN, LABEL_COLUMN, METRICS_DIR, FIGURES_DIR


MANUAL_TEST_PATH = Path("data/manual/manual_test_v2.csv")
MODEL_PATHS = {
    "berturk_v1": Path("models/berturk"),
    "berturk_v2": Path("models/berturk_v2"),
    "berturk_v3": Path("models/berturk_v3")
}
BATCH_SIZE = 32


class ManualEmotionDataset(Dataset):
    def __init__(self, frame, tokenizer, label2id, max_length):
        self.texts = frame[TEXT_COLUMN].astype(str).tolist()
        self.labels = frame[LABEL_COLUMN].map(label2id).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
            "text": self.texts[index]
        }


def load_info(model_dir):
    with open(model_dir / "model_info.json", "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_one_model(model_name, model_dir, frame):
    info = load_info(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    label2id = info["label2id"]
    id2label = {int(key): value for key, value in info["id2label"].items()}
    max_length = info.get("max_length", 96)

    dataset = ManualEmotionDataset(frame, tokenizer, label2id, max_length)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    predictions = []
    actuals = []
    rows = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            probabilities = torch.softmax(outputs.logits, dim=1)
            batch_predictions = torch.argmax(probabilities, dim=1)

            for index in range(len(batch_predictions)):
                predicted_id = int(batch_predictions[index].detach().cpu().item())
                actual_id = int(labels[index].detach().cpu().item())
                confidence = float(probabilities[index][predicted_id].detach().cpu().item())

                predictions.append(predicted_id)
                actuals.append(actual_id)

                rows.append({
                    "model": model_name,
                    "text": batch["text"][index],
                    "actual_label": id2label[actual_id],
                    "predicted_label": id2label[predicted_id],
                    "confidence": confidence,
                    "correct": id2label[actual_id] == id2label[predicted_id]
                })

    labels = sorted(list(id2label.keys()))
    target_names = [id2label[index] for index in labels]

    accuracy = accuracy_score(actuals, predictions)
    macro_f1 = f1_score(actuals, predictions, average="macro")
    weighted_f1 = f1_score(actuals, predictions, average="weighted")

    report = classification_report(
        actuals,
        predictions,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(actuals, predictions, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=target_names)

    fig, ax = plt.subplots(figsize=(9, 7))
    display.plot(ax=ax, xticks_rotation=45, values_format="d")
    plt.tight_layout()
    fig_path = FIGURES_DIR / f"{model_name}_manual_test_confusion_matrix.png"
    plt.savefig(fig_path, dpi=220)
    plt.close()

    result = {
        "model": model_name,
        "manual_test_size": len(frame),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }

    return result, report, rows


def main():
    frame = pd.read_csv(MANUAL_TEST_PATH)
    frame = frame[[TEXT_COLUMN, LABEL_COLUMN]].dropna().reset_index(drop=True)

    all_results = []
    all_reports = {}
    all_predictions = []

    for model_name, model_dir in MODEL_PATHS.items():
        result, report, rows = evaluate_one_model(model_name, model_dir, frame)
        all_results.append(result)
        all_reports[model_name] = report
        all_predictions.extend(rows)

        print()
        print(model_name)
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Macro F1: {result['macro_f1']:.4f}")
        print(f"Weighted F1: {result['weighted_f1']:.4f}")

    results_frame = pd.DataFrame(all_results).sort_values(by="macro_f1", ascending=False)
    predictions_frame = pd.DataFrame(all_predictions)

    results_frame.to_csv(METRICS_DIR / "manual_test_model_comparison.csv", index=False, encoding="utf-8-sig")
    predictions_frame.to_csv(METRICS_DIR / "manual_test_predictions.csv", index=False, encoding="utf-8-sig")

    with open(METRICS_DIR / "manual_test_classification_reports.json", "w", encoding="utf-8") as file:
        json.dump(all_reports, file, ensure_ascii=False, indent=4)

    print()
    print("Manuel test karşılaştırması tamamlandı.")
    print()
    print(results_frame)

    wrong_predictions = predictions_frame[predictions_frame["correct"] == False]
    wrong_predictions.to_csv(METRICS_DIR / "manual_test_wrong_predictions.csv", index=False, encoding="utf-8-sig")

    print()
    print(f"Hatalı tahmin sayısı: {len(wrong_predictions)}")
    print(f"Sonuç dosyası: {METRICS_DIR / 'manual_test_model_comparison.csv'}")
    print(f"Hatalı tahminler: {METRICS_DIR / 'manual_test_wrong_predictions.csv'}")


if __name__ == "__main__":
    main()
