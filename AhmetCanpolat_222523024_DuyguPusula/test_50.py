from src.predict_berturk import get_analyzer

test_cumleleri = [
    ("Bugün sonunda istediğim bölümü kazandım.", "Mutluluk"),
    ("Ailemle güzel bir akşam yemeği yedik.", "Mutluluk"),
    ("Uzun zamandır beklediğim haber geldi.", "Mutluluk"),
    ("Arkadaşlarımla buluşunca çok keyiflendim.", "Mutluluk"),
    ("Memleketime dönmek beni çok mutlu etti.", "Mutluluk"),
    ("Sınavdan yüksek not aldığımı öğrendim.", "Mutluluk"),
    ("Yeni işe kabul edildiğimi duyunca sevindim.", "Mutluluk"),
    ("Bugün kendimi huzurlu ve iyi hissediyorum.", "Mutluluk"),
    ("Doğum günümde herkes beni hatırladı.", "Mutluluk"),
    ("Uzun süredir istediğim bilgisayarı aldım.", "Mutluluk"),

    ("Çok sevdiğim birini kaybettim.", "Üzüntü"),
    ("Arkadaşlarım artık benimle eskisi gibi konuşmuyor.", "Üzüntü"),
    ("Üniversitem bitti, dostlarımdan ayrılacağım.", "Üzüntü"),
    ("Bugün kendimi çok yalnız hissediyorum.", "Üzüntü"),
    ("Planladığım tatil param olmadığı için iptal oldu.", "Üzüntü"),
    ("Sınavdan kaldığımı öğrenince moralim bozuldu.", "Üzüntü"),
    ("Sevdiğim kişi bana eskisi gibi davranmıyor.", "Üzüntü"),
    ("Bütün emeklerim boşa gitmiş gibi hissediyorum.", "Üzüntü"),
    ("Ailemden uzak kalmak beni çok yoruyor.", "Üzüntü"),
    ("Eski günleri düşününce içim burkuluyor.", "Üzüntü"),

    ("Bana böyle davranmasına gerçekten sinirlendim.", "Öfke"),
    ("Bu kadar haksızlık yapılması kabul edilemez.", "Öfke"),
    ("Sınavdan bir puanla kalmam saçmalık.", "Öfke"),
    ("Sürekli aynı hatayı yapması beni delirtiyor.", "Öfke"),
    ("Kimsenin sözümü dinlememesi sinirimi bozuyor.", "Öfke"),
    ("Beni suçlamasına çok öfkelendim.", "Öfke"),
    ("Saatlerce bekletilip hiçbir açıklama yapılmadı.", "Öfke"),
    ("Bu yapılan şey gerçekten saygısızlık.", "Öfke"),
    ("Hakkım yenildiği için çok kızgınım.", "Öfke"),
    ("Bana yalan söylediğini öğrenince çıldırdım.", "Öfke"),

    ("Gece yalnız yürürken çok tedirgin oldum.", "Korku"),
    ("Karanlıkta garip bir ses duydum ve ürktüm.", "Korku"),
    ("Yarınki ameliyat sonucu beni endişelendiriyor.", "Korku"),
    ("Deprem olacağını düşününce içime korku düşüyor.", "Korku"),
    ("Sınav sonucum kötü gelirse diye çok kaygılıyım.", "Korku"),
    ("Telefonuma gelen tehdit mesajı beni korkuttu.", "Korku"),
    ("Kaybolduğumu fark edince panik yaptım.", "Korku"),
    ("Köpek bir anda üzerime koşunca çok korktum.", "Korku"),
    ("Geleceğim hakkında büyük bir endişe taşıyorum.", "Korku"),
    ("Uçağa binmeden önce içimi kötü bir korku sardı.", "Korku"),

    ("Bunu hiç beklemiyordum, gerçekten şaşırdım.", "Şaşkınlık"),
    ("Bir anda karşıma çıkınca ne diyeceğimi bilemedim.", "Şaşkınlık"),
    ("Sınavdan bu kadar yüksek almam beni şaşırttı.", "Şaşkınlık"),
    ("O kişinin böyle davranmasına inanamadım.", "Şaşkınlık"),
    ("Kapıyı açınca herkesin bana sürpriz yaptığını gördüm.", "Şaşkınlık"),
    ("Bu sonucun böyle çıkacağını hiç düşünmemiştim.", "Şaşkınlık"),
    ("Arkadaşımın gizlice evlendiğini öğrenince şaşkına döndüm.", "Şaşkınlık"),
    ("Bir anda elektrikler gidince ne olduğunu anlayamadım.", "Şaşkınlık"),
    ("Bu kadar hızlı değişmesine gerçekten hayret ettim.", "Şaşkınlık"),
    ("Beklemediğim anda adımın anons edilmesi beni şaşırttı.", "Şaşkınlık"),
]

def tahmin_al(analyzer, metin):
    if hasattr(analyzer, "predict"):
        return analyzer.predict(metin)
    if hasattr(analyzer, "analyze"):
        return analyzer.analyze(metin)
    if hasattr(analyzer, "predict_emotion"):
        return analyzer.predict_emotion(metin)
    if callable(analyzer):
        return analyzer(metin)
    raise AttributeError("Analyzer içinde uygun tahmin metodu bulunamadı.")

def duygu_al(sonuc):
    if isinstance(sonuc, dict):
        for key in ["ana_duygu", "duygu", "label", "emotion", "prediction", "predicted_label", "main_emotion"]:
            if key in sonuc:
                return str(sonuc[key])
    return str(sonuc)

def normalize(deger):
    deger = str(deger).strip().lower()
    deger = deger.replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c")
    return deger

def main():
    analyzer = get_analyzer()
    dogru = 0

    print("\n50 Cümlelik Duygu Analizi Testi")
    print("=" * 60)

    for i, (metin, beklenen) in enumerate(test_cumleleri, start=1):
        sonuc = tahmin_al(analyzer, metin)
        tahmin = duygu_al(sonuc)

        durum = "DOĞRU" if normalize(beklenen) in normalize(tahmin) else "YANLIŞ"

        if durum == "DOĞRU":
            dogru += 1

        print(f"\n{i}. Cümle: {metin}")
        print(f"Beklenen: {beklenen}")
        print(f"Tahmin: {tahmin}")
        print(f"Durum: {durum}")

    oran = dogru / len(test_cumleleri) * 100

    print("\n" + "=" * 60)
    print(f"Toplam doğru: {dogru}/{len(test_cumleleri)}")
    print(f"Başarı oranı: %{oran:.2f}")

if __name__ == "__main__":
    main()
