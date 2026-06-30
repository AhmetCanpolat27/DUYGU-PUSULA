import json
import numpy as np
import joblib

from src.config import CLASSICAL_MODEL_DIR, METRICS_DIR
from src.preprocessing import normalize_text


MODEL_PATH = CLASSICAL_MODEL_DIR / "best_classical_model.joblib"
INFO_PATH = METRICS_DIR / "best_classical_model.json"

DISPLAY_LABELS = {
    "mutluluk": "Mutluluk",
    "uzuntu": "Üzüntü",
    "ofke": "Öfke",
    "korku": "Korku",
    "saskinlik": "Şaşkınlık"
}


def load_model():
    model = joblib.load(MODEL_PATH)

    with open(INFO_PATH, "r", encoding="utf-8") as file:
        info = json.load(file)

    return model, info


def get_scores(model, text):
    clean_text = normalize_text(text)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([clean_text])[0]
        classes = model.classes_
        return dict(zip(classes, probabilities))

    decision_values = model.decision_function([clean_text])[0]
    exp_values = np.exp(decision_values - np.max(decision_values))
    probabilities = exp_values / exp_values.sum()
    classes = model.classes_
    return dict(zip(classes, probabilities))


def predict_emotion(text):
    model, info = load_model()
    scores = get_scores(model, text)

    best_label = max(scores, key=scores.get)
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    return {
        "text": text,
        "model": info["best_model"],
        "label": best_label,
        "display_label": DISPLAY_LABELS.get(best_label, best_label),
        "confidence": float(scores[best_label]),
        "scores": [
            {
                "label": label,
                "display_label": DISPLAY_LABELS.get(label, label),
                "score": float(score)
            }
            for label, score in sorted_scores
        ]
    }


def print_prediction(result):
    print()
    print("Girilen metin:")
    print(result["text"])
    print()
    print(f"Tahmin edilen duygu: {result['display_label']}")
    print(f"Güven skoru: %{result['confidence'] * 100:.2f}")
    print(f"Kullanılan model: {result['model']}")
    print()
    print("Duygu olasılıkları:")

    for item in result["scores"]:
        print(f"{item['display_label']}: %{item['score'] * 100:.2f}")


if __name__ == "__main__":
    text = input("Metin gir: ").strip()

    if not text:
        print("Metin boş olamaz.")
    else:
        result = predict_emotion(text)
        print_prediction(result)
