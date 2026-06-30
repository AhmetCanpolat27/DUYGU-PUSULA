import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = Path("models/berturk")


class BerturkEmotionAnalyzer:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model klasörü bulunamadı: {self.model_dir}")

        with open(self.model_dir / "model_info.json", "r", encoding="utf-8") as file:
            self.info = json.load(file)

        self.id2label = {int(key): value for key, value in self.info["id2label"].items()}
        self.max_length = self.info.get("max_length", 96)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        encoded = self.tokenizer(
            str(text),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        probabilities = torch.softmax(outputs.logits, dim=1).squeeze(0).detach().cpu()

        scores = []
        for index, score in enumerate(probabilities):
            scores.append({
                "label": self.id2label[index],
                "score": float(score)
            })

        scores = sorted(scores, key=lambda item: item["score"], reverse=True)

        first = scores[0]
        second = scores[1]
        gap = first["score"] - second["score"]

        if first["score"] >= 0.80 and gap >= 0.25:
            confidence_level = "Yüksek"
        elif first["score"] >= 0.55:
            confidence_level = "Orta"
        else:
            confidence_level = "Düşük"

        if second["score"] >= 0.25 and gap <= 0.20:
            decision_type = "Karma duygu olabilir"
        elif first["score"] < 0.50:
            decision_type = "Belirsiz"
        else:
            decision_type = "Baskın duygu"

        return {
            "text": text,
            "label": first["label"],
            "score": first["score"],
            "second_label": second["label"],
            "second_score": second["score"],
            "confidence_level": confidence_level,
            "decision_type": decision_type,
            "all_scores": scores
        }


_ANALYZER = None


def get_analyzer():
    global _ANALYZER

    if _ANALYZER is None:
        _ANALYZER = BerturkEmotionAnalyzer()

    return _ANALYZER


