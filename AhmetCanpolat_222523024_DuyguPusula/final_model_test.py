from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.predict_berturk import get_analyzer

TEST_DATA = [
    ("Bugün beklediğim güzel haber geldi ve kendimi gerçekten iyi hissediyorum.", "mutluluk"),
    ("Ailemle geçirdiğim bu sakin gün bana çok huzur verdi.", "mutluluk"),
    ("Uzun süredir istediğim şeyi başarmak beni çok sevindirdi.", "mutluluk"),
    ("Arkadaşlarımla yeniden bir araya gelmek bana iyi geldi.", "mutluluk"),
    ("Bu sonucu görünce emeklerimin boşa gitmediğini anladım.", "mutluluk"),
    ("Yeni bir başlangıç yapacağım için içimde güzel bir heyecan var.", "mutluluk"),
    ("Bugün her şey yolunda gittiği için kendimi çok rahat hissediyorum.", "mutluluk"),
    ("Sevdiğim insanların yanımda olduğunu görmek beni mutlu etti.", "mutluluk"),
    ("Uzun zamandır ilk defa bu kadar huzurlu uyandım.", "mutluluk"),
    ("Aldığım güzel mesaj bütün günümü güzelleştirdi.", "mutluluk"),

    ("Bugün kendimi çok yalnız ve kırgın hissediyorum.", "uzuntu"),
    ("Beklediğim desteği görememek beni derinden üzdü.", "uzuntu"),
    ("Sevdiğim insanlardan uzak kalmak içimi burkuyor.", "uzuntu"),
    ("Bütün çabama rağmen başarısız olmak moralimi bozdu.", "uzuntu"),
    ("Eski günleri hatırladıkça içimde bir boşluk oluşuyor.", "uzuntu"),
    ("Arkadaşlarımla artık eskisi gibi konuşamamak beni üzüyor.", "uzuntu"),
    ("Bu vedanın bu kadar zor olacağını düşünmemiştim.", "uzuntu"),
    ("Hayal ettiğim şeylerin gerçekleşmemesi beni kırdı.", "uzuntu"),
    ("Bugün hiçbir şey yapmak istemeyecek kadar mutsuzum.", "uzuntu"),
    ("Onun artık hayatımda olmaması içimi acıtıyor.", "uzuntu"),

    ("Bana haksızlık yapılması gerçekten sinirimi bozdu.", "ofke"),
    ("Sürekli aynı hatayı yapmaları beni çok kızdırıyor.", "ofke"),
    ("Bu kadar emek verdikten sonra suçlanmak kabul edilemez.", "ofke"),
    ("Beni dinlemeden karar vermelerine çok öfkelendim.", "ofke"),
    ("Saatlerce bekletilip açıklama yapılmaması sinir bozucu.", "ofke"),
    ("Bana yalan söylendiğini öğrenince çok kızdım.", "ofke"),
    ("Bu davranışın saygısızlık olduğunu düşünüyorum.", "ofke"),
    ("Hakkımın yenildiğini görmek beni öfkelendirdi.", "ofke"),
    ("Her şeyi benim üzerime yıkmaları artık canımı sıkıyor.", "ofke"),
    ("Bu kadar sorumsuz davranmaları beni gerçekten delirtiyor.", "ofke"),

    ("Gece yalnız yürürken arkamdan gelen sesler beni tedirgin etti.", "korku"),
    ("Bir anda yüksek bir ses duyunca korkudan yerimden sıçradım.", "korku"),
    ("Yarınki sonucu düşünmek beni ciddi şekilde endişelendiriyor.", "korku"),
    ("Karanlıkta birinin beni takip ettiğini sandım.", "korku"),
    ("Kaybolduğumu fark edince panik yapmaya başladım.", "korku"),
    ("Telefonuma gelen tehdit mesajı beni çok korkuttu.", "korku"),
    ("Kötü bir şey olacakmış gibi hissetmek beni huzursuz ediyor.", "korku"),
    ("Hastaneden gelecek haberi beklerken içimi korku kapladı.", "korku"),
    ("Köpeğin üzerime doğru koşması beni çok ürküttü.", "korku"),
    ("Kontrolü kaybettiğimi hissedince panikledim.", "korku"),

    ("Bu sonucu hiç beklemiyordum ve gerçekten çok şaşırdım.", "saskinlik"),
    ("Bir anda ismimin anons edilmesi beni şaşkına çevirdi.", "saskinlik"),
    ("Onun böyle bir karar vermesine inanamadım.", "saskinlik"),
    ("Karşıma beklenmedik biri çıkınca ne diyeceğimi bilemedim.", "saskinlik"),
    ("Bu kadar kısa sürede bu kadar değişmesine hayret ettim.", "saskinlik"),
    ("Kapıyı açınca herkesin bana sürpriz yaptığını gördüm.", "saskinlik"),
    ("Duyduğum haber karşısında birkaç saniye konuşamadım.", "saskinlik"),
    ("Sonucun tamamen ters çıkması beni şaşkınlık içinde bıraktı.", "saskinlik"),
    ("Hiç tahmin etmediğim bir anda böyle bir teklif aldım.", "saskinlik"),
    ("Bunun gerçek olduğuna ilk başta inanamadım.", "saskinlik"),
]

LABELS = ["mutluluk", "uzuntu", "ofke", "korku", "saskinlik"]

DISPLAY_LABELS = {
    "mutluluk": "Mutluluk",
    "uzuntu": "Üzüntü",
    "ofke": "Öfke",
    "korku": "Korku",
    "saskinlik": "Şaşkınlık"
}

def normalize_label(value):
    value = str(value).strip().lower()
    value = value.replace("ı", "i")
    value = value.replace("ğ", "g")
    value = value.replace("ü", "u")
    value = value.replace("ş", "s")
    value = value.replace("ö", "o")
    value = value.replace("ç", "c")
    return value

def predict_with_analyzer(analyzer, text):
    if hasattr(analyzer, "predict"):
        return analyzer.predict(text)
    if hasattr(analyzer, "analyze"):
        return analyzer.analyze(text)
    if hasattr(analyzer, "predict_emotion"):
        return analyzer.predict_emotion(text)
    if callable(analyzer):
        return analyzer(text)
    raise AttributeError("Tahmin metodu bulunamadı.")

def extract_prediction(result):
    if isinstance(result, dict):
        for key in ["ana_duygu", "duygu", "label", "emotion", "prediction", "predicted_label", "main_emotion"]:
            if key in result:
                return normalize_label(result[key])

        scores = (
            result.get("skorlar")
            or result.get("scores")
            or result.get("olasiliklar")
            or result.get("probabilities")
            or result.get("class_scores")
            or result.get("tum_skorlar")
            or {}
        )

        if isinstance(scores, dict) and len(scores) > 0:
            best_label = max(scores.items(), key=lambda x: float(x[1]))[0]
            return normalize_label(best_label)

    return normalize_label(result)

def main():
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)

    analyzer = get_analyzer()

    y_true = []
    y_pred = []
    rows = []

    print("\nDuyguPusula Final Model Testi")
    print("=" * 70)

    for i, (text, expected) in enumerate(TEST_DATA, start=1):
        result = predict_with_analyzer(analyzer, text)
        predicted = extract_prediction(result)

        expected = normalize_label(expected)
        status = "DOĞRU" if expected == predicted else "YANLIŞ"

        y_true.append(expected)
        y_pred.append(predicted)

        rows.append({
            "no": i,
            "metin": text,
            "beklenen": expected,
            "tahmin": predicted,
            "durum": status
        })

        print(f"\n{i}. Metin: {text}")
        print(f"Beklenen: {DISPLAY_LABELS.get(expected, expected)}")
        print(f"Tahmin: {DISPLAY_LABELS.get(predicted, predicted)}")
        print(f"Durum: {status}")

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=[DISPLAY_LABELS[label] for label in LABELS],
        digits=4,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    df_results = pd.DataFrame(rows)
    df_results.to_csv(output_dir / "final_test_results.csv", index=False, encoding="utf-8-sig")

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("DuyguPusula Final Model Testi\n")
        f.write("=" * 70)
        f.write("\n\n")
        f.write(f"Toplam test cümlesi: {len(TEST_DATA)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xticks(range(len(LABELS)), [DISPLAY_LABELS[label] for label in LABELS], rotation=30)
    plt.yticks(range(len(LABELS)), [DISPLAY_LABELS[label] for label in LABELS])
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    print("\n" + "=" * 70)
    print(f"Toplam test cümlesi: {len(TEST_DATA)}")
    print(f"Doğruluk oranı: %{accuracy * 100:.2f}")
    print("\nSınıf bazlı rapor:")
    print(report)

    print("\nKaydedilen dosyalar:")
    print(output_dir / "final_test_results.csv")
    print(output_dir / "classification_report.txt")
    print(output_dir / "confusion_matrix.png")

if __name__ == "__main__":
    main()
