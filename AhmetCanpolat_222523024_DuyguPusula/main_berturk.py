from src.predict_berturk import get_analyzer


DISPLAY_NAMES = {
    "mutluluk": "Mutluluk",
    "uzuntu": "Üzüntü",
    "ofke": "Öfke",
    "korku": "Korku",
    "saskinlik": "Şaşkınlık"
}


def format_label(label):
    return DISPLAY_NAMES.get(label, label)


def main():
    print("BERTurk v2 Türkçe Çok Sınıflı Duygu Analizi")
    print("Çıkmak için q yaz.")
    print()
    print("Model yükleniyor...")

    analyzer = get_analyzer()

    print("Model hazır.")
    print()

    while True:
        text = input("Metin gir: ").strip()

        if text.lower() in {"q", "quit", "exit", "çık"}:
            break

        if not text:
            continue

        result = analyzer.predict(text)

        print()
        print("Tahmin sonucu")
        print("-" * 40)
        print(f"Ana duygu: {format_label(result['label'])}")
        print(f"Güven: %{result['score'] * 100:.2f}")
        print(f"İkinci olası duygu: {format_label(result['second_label'])} (%{result['second_score'] * 100:.2f})")
        print(f"Güven seviyesi: {result['confidence_level']}")
        print(f"Karar tipi: {result['decision_type']}")
        print()
        print("Tüm sınıf skorları:")

        for item in result["all_scores"]:
            print(f"- {format_label(item['label'])}: %{item['score'] * 100:.2f}")

        print()


if __name__ == "__main__":
    main()

