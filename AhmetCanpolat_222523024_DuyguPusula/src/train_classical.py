import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from src.config import (
    PROCESSED_DATA_PATH,
    TEXT_COLUMN,
    LABEL_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    CLASSICAL_MODEL_DIR,
    METRICS_DIR,
    FIGURES_DIR
)
from src.preprocessing import normalize_text


def load_data():
    frame = pd.read_csv(PROCESSED_DATA_PATH)
    frame = frame[[TEXT_COLUMN, LABEL_COLUMN]].dropna()
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].apply(normalize_text)
    frame = frame[frame[TEXT_COLUMN].str.len() > 2]
    frame = frame.reset_index(drop=True)
    return frame


def build_models():
    return {
        "naive_bayes": MultinomialNB(alpha=0.5),
        "complement_nb": ComplementNB(alpha=0.5),
        "logistic_regression": LogisticRegression(max_iter=3000, class_weight="balanced", C=2.0),
        "linear_svm": LinearSVC(class_weight="balanced", C=1.0)
    }


def build_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )),
        ("model", model)
    ])


def save_confusion_matrix(y_true, y_pred, labels, model_name):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    display.plot(ax=ax, xticks_rotation=45, values_format="d")
    plt.tight_layout()
    path = FIGURES_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def train():
    frame = load_data()

    label_counts = frame[LABEL_COLUMN].value_counts()
    valid_labels = label_counts[label_counts >= 2].index.tolist()
    frame = frame[frame[LABEL_COLUMN].isin(valid_labels)].reset_index(drop=True)

    x = frame[TEXT_COLUMN]
    y = frame[LABEL_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    labels = sorted(y.unique())
    models = build_models()
    results = []
    reports = {}
    best_name = None
    best_score = -1
    best_pipeline = None

    for model_name, estimator in models.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(y_test, predictions, average="macro")
        weighted_f1 = f1_score(y_test, predictions, average="weighted")

        results.append({
            "model": model_name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1
        })

        reports[model_name] = classification_report(
            y_test,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0
        )

        save_confusion_matrix(y_test, predictions, labels, model_name)

        if macro_f1 > best_score:
            best_score = macro_f1
            best_name = model_name
            best_pipeline = pipeline

        print(f"{model_name} tamamlandı")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print()

    results_frame = pd.DataFrame(results).sort_values(by="macro_f1", ascending=False)
    results_path = METRICS_DIR / "classical_model_results.csv"
    reports_path = METRICS_DIR / "classical_classification_reports.json"
    best_info_path = METRICS_DIR / "best_classical_model.json"
    best_model_path = CLASSICAL_MODEL_DIR / "best_classical_model.joblib"

    results_frame.to_csv(results_path, index=False, encoding="utf-8-sig")

    with open(reports_path, "w", encoding="utf-8") as file:
        json.dump(reports, file, ensure_ascii=False, indent=4)

    with open(best_info_path, "w", encoding="utf-8") as file:
        json.dump({
            "best_model": best_name,
            "best_macro_f1": best_score,
            "labels": labels,
            "train_size": len(x_train),
            "test_size": len(x_test),
            "total_size": len(frame)
        }, file, ensure_ascii=False, indent=4)

    joblib.dump(best_pipeline, best_model_path)

    print("Klasik model eğitimi tamamlandı.")
    print()
    print(results_frame)
    print()
    print(f"En iyi model: {best_name}")
    print(f"Model kaydedildi: {best_model_path}")
    print(f"Sonuçlar kaydedildi: {results_path}")


if __name__ == "__main__":
    train()
