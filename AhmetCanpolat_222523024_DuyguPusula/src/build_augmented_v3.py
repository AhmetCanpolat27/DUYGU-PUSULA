import pandas as pd
from pathlib import Path

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

AUGMENTED_V2_PATH = Path("data/processed/turkish_tweet_emotion_augmented_v2.csv")
AUGMENTED_V3_PATH = Path("data/processed/turkish_tweet_emotion_augmented_v3.csv")
HARD_EXTRA_PATH = Path("data/manual/hard_extra_v3.csv")

EXAMPLES = [
    ("Aradığım nadir kitabı kütüphanede bulunca uzun süredir beklediğim şeye kavuşmuş gibi hissettim.", "mutluluk"),
    ("Yıllardır peşinde olduğum eski dergi sayısını sahafın rafında görünce içim sevinçle doldu.", "mutluluk"),
    ("Aylardır aradığım plak albümünü sonunda bulmak bütün günümü güzelleştirdi.", "mutluluk"),
    ("Kayıp sandığım defterimi çekmecenin arkasında bulunca büyük bir rahatlama yaşadım.", "mutluluk"),
    ("Uzun süredir almak istediğim eğitim kabulünü görünce emeğimin karşılığını aldığımı hissettim.", "mutluluk"),
    ("Proje dosyamın yedeğini bulunca üzerimden büyük bir yük kalktı.", "mutluluk"),

    ("Eski sandığın içinden hiç beklenmeyen tarihi belgeler çıkınca bir süre neye baktığımı anlayamadım.", "saskinlik"),
    ("Bahçeyi kazarken toprağın altından eski mühürlü bir kutu çıkması hepimizi şaşkına çevirdi.", "saskinlik"),
    ("Radyoda kendi adımı duyunca birkaç saniye bunun gerçek olup olmadığını anlamaya çalıştım.", "saskinlik"),
    ("Müzede hareketsiz sandığım mekanik heykelin bir anda dönmeye başlaması beni afallattı.", "saskinlik"),
    ("Yıllardır sıradan sandığımız eşyanın antika değerinde olduğunu öğrenince donup kaldık.", "saskinlik"),
    ("Beklemediğim bir anda ekranda kendi tasarımımın uluslararası bir yarışmada gösterildiğini gördüm.", "saskinlik"),
    ("Eski evin duvarının arkasından gizli bir oda çıkınca herkes gözlerine inanamadı.", "saskinlik"),
    ("Sessiz sakin bildiğimiz görevlinin kalabalığın önünde profesyonelce dans etmesi herkesi şaşırttı.", "saskinlik"),
    ("Aynaya baktığımda saç rengimdeki ani değişimi fark edince birkaç saniye kendime gelemedim.", "saskinlik"),
    ("Kutunun içinden beklediğim parçalar yerine tamamen farklı ve eski mühürlü objeler çıkması tuhaftı.", "saskinlik"),

    ("Telefonuma her gün onlarca spam mesaj gelmesi artık sabrımı taşırıyor.", "ofke"),
    ("Sürekli bahis mesajı gönderen numaraları engellesem de yenilerinin gelmesi beni çıldırtıyor.", "ofke"),
    ("İstemediğim reklam mesajlarını defalarca iptal etmeme rağmen göndermeye devam etmeleri sinir bozucu.", "ofke"),
    ("Gün içinde durmadan gelen sahte kampanya mesajları yüzünden telefonumu kullanamaz oldum.", "ofke"),
    ("Numaramı izinsiz şekilde reklam listelerine eklemelerine gerçekten öfkeliyim.", "ofke"),
    ("Engellediğim halde farklı numaralardan tekrar tekrar mesaj atmaları artık saygısızlık.", "ofke"),
    ("Her sabah aynı dolandırıcı mesajları görmek beni ciddi şekilde rahatsız ediyor.", "ofke"),
    ("Spam mesaj atan firmaların hiçbir sorumluluk almaması kabul edilemez.", "ofke"),

    ("Bütün gün üzerime çöken gri hava ve sessizlik içimi ağırlaştırdı.", "uzuntu"),
    ("Şehrin soğuk ve renksiz hali bugün içimde tarifsiz bir boşluk bıraktı.", "uzuntu"),
    ("Kalabalığın ortasında bile kimseyle paylaşamadığım bir yalnızlık hissettim.", "uzuntu"),
    ("Güzel haberi almış olsam da yanımda sevincimi paylaşacak kimsenin olmaması içimi burktu.", "uzuntu"),
    ("Başarımı kutlamak istedim ama etrafımda arayabileceğim kimse olmadığını fark ettim.", "uzuntu"),
    ("Son dakikada gelen kötü golle bütün umutlarımız bir anda söndü.", "uzuntu"),
    ("Maçın bitimine saniyeler kala elenmek bütün takımın moralini yerle bir etti.", "uzuntu"),
    ("Aylarca beklediğim sonucun böyle bitmesi içimde büyük bir kırgınlık bıraktı.", "uzuntu"),

    ("Eski duvardaki figürlerin hareket ettiğini sandığımda önce ne gördüğümü anlayamadım.", "saskinlik"),
    ("Tarihi binanın tavanındaki desenlerin ışıkla değişiyormuş gibi görünmesi beni şaşırttı.", "saskinlik"),
    ("Müzede sergilenen eski sandalyenin kendi kendine kıpırdadığını sanınca afalladım.", "saskinlik"),
    ("Bir objenin normalden farklı hareket ettiğini görünce bunun nasıl mümkün olduğunu düşündüm.", "saskinlik"),

    ("Sisli sokakta uzaktan gelen belirsiz ayak sesleri beni tedirgin etti.", "korku"),
    ("Gece yarısı kapının kolunun yavaşça hareket ettiğini duyunca nefesimi tuttum.", "korku"),
    ("Telefonuma gelen giriş uyarısında tanımadığım bir şehir görünce paniğe kapıldım.", "korku"),
    ("Banka hesabıma yabancı bir cihazdan erişildiğini görünce elim ayağım titredi.", "korku")
]


def main():
    base = pd.read_csv(AUGMENTED_V2_PATH)
    extra = pd.DataFrame(EXAMPLES, columns=[TEXT_COLUMN, LABEL_COLUMN])

    if "raw_label" in base.columns:
        extra["raw_label"] = extra[LABEL_COLUMN]
        extra = extra[[TEXT_COLUMN, "raw_label", LABEL_COLUMN]]
        base = base[[TEXT_COLUMN, "raw_label", LABEL_COLUMN]]
    else:
        base = base[[TEXT_COLUMN, LABEL_COLUMN]]
        extra = extra[[TEXT_COLUMN, LABEL_COLUMN]]

    merged = pd.concat([base, extra], ignore_index=True)
    merged["text_key"] = merged[TEXT_COLUMN].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    merged = merged.drop_duplicates(subset=["text_key", LABEL_COLUMN])
    merged = merged.drop(columns=["text_key"])
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    HARD_EXTRA_PATH.parent.mkdir(parents=True, exist_ok=True)
    extra.to_csv(HARD_EXTRA_PATH, index=False, encoding="utf-8-sig")
    merged.to_csv(AUGMENTED_V3_PATH, index=False, encoding="utf-8-sig")

    print("Hard extra v3 oluşturuldu.")
    print(f"Ek örnek sayısı: {len(extra)}")
    print(f"V3 toplam satır: {len(merged)}")
    print()
    print("Ek veri dağılımı:")
    print(extra[LABEL_COLUMN].value_counts())
    print()
    print("V3 toplam dağılım:")
    print(merged[LABEL_COLUMN].value_counts())


if __name__ == "__main__":
    main()
