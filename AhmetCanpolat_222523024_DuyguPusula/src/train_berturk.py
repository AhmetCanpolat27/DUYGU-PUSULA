import json
import random
import time

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm.auto import tqdm

from src.config import (
    PROCESSED_DATA_PATH,
    TEXT_COLUMN,
    LABEL_COLUMN,
    RANDOM_STATE,
    BERTURK_MODEL_DIR,
    METRICS_DIR,
    FIGURES_DIR
)


MODEL_NAME = "dbmdz/bert-base-turkish-cased"
MAX_LENGTH = 96
BATCH_SIZE = 24
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
PATIENCE = 2


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            str(self.texts[index]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long)
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_text(text):
    value = str(text).strip()
    value = value.replace("\n", " ").replace("\r", " ")
    value = " ".join(value.split())
    return value


def load_frame():
    frame = pd.read_csv(PROCESSED_DATA_PATH)
    frame = frame[[TEXT_COLUMN, LABEL_COLUMN]].dropna()
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].apply(prepare_text)
    frame = frame[frame[TEXT_COLUMN].str.len() > 2]
    frame = frame.reset_index(drop=True)
    return frame


def create_splits(frame):
    labels = sorted(frame[LABEL_COLUMN].unique())
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    frame["label_id"] = frame[LABEL_COLUMN].map(label2id)

    train_frame, temp_frame = train_test_split(
        frame,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=frame["label_id"]
    )

    valid_frame, test_frame = train_test_split(
        temp_frame,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_frame["label_id"]
    )

    return train_frame.reset_index(drop=True), valid_frame.reset_index(drop=True), test_frame.reset_index(drop=True), label2id, id2label


def build_loader(frame, tokenizer, shuffle):
    dataset = EmotionDataset(frame[TEXT_COLUMN], frame["label_id"], tokenizer, MAX_LENGTH)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )


def train_epoch(model, loader, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0

    progress = tqdm(loader, desc="Eğitim", leave=False)

    for batch in progress:
        optimizer.zero_grad(set_to_none=True)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(len(loader), 1)


def evaluate_model(model, loader, device):
    model.eval()

    total_loss = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

            total_loss += outputs.loss.item()
            batch_predictions = torch.argmax(outputs.logits, dim=1)

            predictions.extend(batch_predictions.detach().cpu().numpy().tolist())
            actuals.extend(labels.detach().cpu().numpy().tolist())

    accuracy = accuracy_score(actuals, predictions)
    macro_f1 = f1_score(actuals, predictions, average="macro")
    weighted_f1 = f1_score(actuals, predictions, average="weighted")

    return total_loss / max(len(loader), 1), accuracy, macro_f1, weighted_f1, predictions, actuals


def save_confusion_matrix(actuals, predictions, labels, id2label):
    target_names = [id2label[index] for index in range(len(labels))]
    matrix = confusion_matrix(actuals, predictions, labels=list(range(len(labels))))

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=target_names)

    fig, ax = plt.subplots(figsize=(9, 7))
    display.plot(ax=ax, xticks_rotation=45, values_format="d")
    plt.tight_layout()

    path = FIGURES_DIR / "berturk_confusion_matrix.png"
    plt.savefig(path, dpi=220)
    plt.close()

    return path


def save_history_plot(history):
    history_frame = pd.DataFrame(history)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(history_frame["epoch"], history_frame["train_loss"], marker="o", label="train_loss")
    ax.plot(history_frame["epoch"], history_frame["valid_loss"], marker="o", label="valid_loss")
    ax.plot(history_frame["epoch"], history_frame["valid_macro_f1"], marker="o", label="valid_macro_f1")
    ax.set_xlabel("epoch")
    ax.legend()
    plt.tight_layout()

    path = FIGURES_DIR / "berturk_training_history.png"
    plt.savefig(path, dpi=220)
    plt.close()

    return path


def save_model(model, tokenizer, label2id, id2label, metrics):
    BERTURK_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(BERTURK_MODEL_DIR)
    tokenizer.save_pretrained(BERTURK_MODEL_DIR)

    info = {
        "base_model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "labels": list(label2id.keys()),
        "label2id": label2id,
        "id2label": {str(key): value for key, value in id2label.items()},
        "metrics": metrics
    }

    with open(BERTURK_MODEL_DIR / "model_info.json", "w", encoding="utf-8") as file:
        json.dump(info, file, ensure_ascii=False, indent=4)


def train():
    start_time = time.time()
    set_seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Cihaz: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    frame = load_frame()
    train_frame, valid_frame, test_frame, label2id, id2label = create_splits(frame)

    print(f"Toplam veri: {len(frame)}")
    print(f"Eğitim: {len(train_frame)}")
    print(f"Doğrulama: {len(valid_frame)}")
    print(f"Test: {len(test_frame)}")
    print(f"Sınıflar: {list(label2id.keys())}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label={index: label for label, index in label2id.items()},
        label2id=label2id
    )

    model.to(device)

    train_loader = build_loader(train_frame, tokenizer, True)
    valid_loader = build_loader(valid_frame, tokenizer, False)
    test_loader = build_loader(test_frame, tokenizer, False)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_valid_f1 = -1
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        print()
        print(f"Epoch {epoch}/{EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        valid_loss, valid_accuracy, valid_macro_f1, valid_weighted_f1, _, _ = evaluate_model(model, valid_loader, device)

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_accuracy": valid_accuracy,
            "valid_macro_f1": valid_macro_f1,
            "valid_weighted_f1": valid_weighted_f1
        }

        history.append(history_row)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}")
        print(f"Valid Accuracy: {valid_accuracy:.4f}")
        print(f"Valid Macro F1: {valid_macro_f1:.4f}")

        if valid_macro_f1 > best_valid_f1:
            best_valid_f1 = valid_macro_f1
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_accuracy, test_macro_f1, test_weighted_f1, test_predictions, test_actuals = evaluate_model(model, test_loader, device)

    labels = list(label2id.keys())
    target_names = [id2label[index] for index in range(len(labels))]

    report = classification_report(
        test_actuals,
        test_predictions,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "model": "berturk",
        "base_model": MODEL_NAME,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "accuracy": test_accuracy,
        "macro_f1": test_macro_f1,
        "weighted_f1": test_weighted_f1,
        "train_size": len(train_frame),
        "valid_size": len(valid_frame),
        "test_size": len(test_frame),
        "total_size": len(frame),
        "device": str(device),
        "duration_minutes": round((time.time() - start_time) / 60, 2)
    }

    save_model(model, tokenizer, label2id, id2label, metrics)
    save_confusion_matrix(test_actuals, test_predictions, labels, id2label)
    save_history_plot(history)

    pd.DataFrame(history).to_csv(METRICS_DIR / "berturk_training_history.csv", index=False, encoding="utf-8-sig")

    with open(METRICS_DIR / "berturk_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=4)

    with open(METRICS_DIR / "berturk_classification_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=4)

    print()
    print("BERTurk eğitimi tamamlandı.")
    print(f"Best Epoch: {best_epoch}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Macro F1: {test_macro_f1:.4f}")
    print(f"Test Weighted F1: {test_weighted_f1:.4f}")
    print(f"Model kaydedildi: {BERTURK_MODEL_DIR}")


if __name__ == "__main__":
    train()
