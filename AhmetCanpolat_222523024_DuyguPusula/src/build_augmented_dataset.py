import pandas as pd

from src.config import PROCESSED_DATA_PATH, TEXT_COLUMN, LABEL_COLUMN


EXAMPLES = [
    ("Bugün sonunda mezun oldum ve ailemle bu anı kutladık.", "mutluluk"),
    ("Aylardır beklediğim işe kabul edildiğimi öğrendim, çok mutluyum.", "mutluluk"),
    ("Sınavdan yüksek aldım, bütün emeklerime değdi.", "mutluluk"),
    ("Uzun zamandır görmediğim arkadaşlarımla buluşmak bana çok iyi geldi.", "mutluluk"),
    ("Hayalini kurduğum bölümü kazandım, sevinçten yerimde duramıyorum.", "mutluluk"),
    ("Siparişim beklediğimden güzel geldi, gerçekten çok memnun kaldım.", "mutluluk"),
    ("Bugün aldığım haber beni çok sevindirdi.", "mutluluk"),
    ("Ailemle geçirdiğim bu güzel gün beni çok mutlu etti.", "mutluluk"),
    ("Başardığımı görmek bana büyük bir sevinç verdi.", "mutluluk"),
    ("Uzun süredir beklediğim güzel haberi sonunda aldım.", "mutluluk"),
    ("Bebeğimizin ilk adımlarını görmek hepimizi çok sevindirdi.", "mutluluk"),
    ("Bu kadar güzel bir sonuç beklemiyordum, çok mutluyum.", "mutluluk"),
    ("Arkadaşımın başarısına çok sevindim.", "mutluluk"),
    ("Bugün kendimi huzurlu ve keyifli hissediyorum.", "mutluluk"),
    ("Her şey yolunda gidiyor, içim rahat ve mutluyum.", "mutluluk"),

    ("Sınavdan düşük aldım ve dersi geçemedim.", "uzuntu"),
    ("Bugün kendimi çok yalnız ve moralsiz hissediyorum.", "uzuntu"),
    ("Sevdiğim insanın benden uzaklaşması beni çok üzdü.", "uzuntu"),
    ("Köpeğimi kaybettim, ev artık bomboş geliyor.", "uzuntu"),
    ("Aylardır beklediğim konser iptal edildi, büyük hayal kırıklığı yaşadım.", "uzuntu"),
    ("Emek verdiğim halde istediğim sonucu alamadım.", "uzuntu"),
    ("Arkadaşım başka bir şehre taşındı ve onu çok özleyeceğim.", "uzuntu"),
    ("Bugün hiçbir şey yapmak gelmiyor içimden.", "uzuntu"),
    ("Beklediğim mesaj gelmeyince moralim çok bozuldu.", "uzuntu"),
    ("Başaramadığımı düşünmek beni çok üzüyor.", "uzuntu"),
    ("Ailemden uzak kalmak beni çok yıpratıyor.", "uzuntu"),
    ("Hayalini kurduğum şey olmayınca çok kırıldım.", "uzuntu"),
    ("Bu kadar çabalayıp kaybetmek beni gerçekten üzdü.", "uzuntu"),
    ("İçimde tarif edemediğim bir boşluk var.", "uzuntu"),
    ("Bugün her şey üstüme geliyormuş gibi hissediyorum.", "uzuntu"),

    ("Siparişim günlerdir kargoya verilmedi, bu ne sorumsuzluk.", "ofke"),
    ("Müşteri hizmetleri telefonları açmıyor, gerçekten sinir bozucu.", "ofke"),
    ("Kutudan kırık ürün çıktı, böyle saçmalık olamaz.", "ofke"),
    ("Paramı aldılar ama hizmet vermediler, buna çok kızgınım.", "ofke"),
    ("Bu kadar bekletilmek gerçekten sinirimi bozdu.", "ofke"),
    ("Haksız yere suçlanmak beni çok öfkelendirdi.", "ofke"),
    ("Kimse sorumluluk almıyor, bu durumdan bıktım.", "ofke"),
    ("Sürekli aynı hatayı yapmaları beni deli ediyor.", "ofke"),
    ("Bu kadar kötü hizmet görmedim, sakın tercih etmeyin.", "ofke"),
    ("Bana verilen sözlerin tutulmaması çok sinir bozucu.", "ofke"),
    ("Durduk yere azar işitmek beni çok kızdırdı.", "ofke"),
    ("Bu yapılan tam bir saygısızlık.", "ofke"),
    ("İnsanları bu şekilde mağdur etmeye hakları yok.", "ofke"),
    ("Böyle ilgisiz bir satıcı görmedim.", "ofke"),
    ("Sürekli oyalıyorlar, artık sabrım kalmadı.", "ofke"),

    ("Gece evden gelen sesler beni çok korkuttu.", "korku"),
    ("Deprem olunca panikten ne yapacağımı bilemedim.", "korku"),
    ("Telefonuma hesabımdan para çekildi mesajı geldi, çok endişelendim.", "korku"),
    ("Karanlık sokakta tek başıma yürürken çok tedirgin oldum.", "korku"),
    ("Birinin beni takip ettiğini sandım ve çok korktum.", "korku"),
    ("Hastane sonucunu beklemek beni çok geriyor.", "korku"),
    ("Uçağın türbülansa girmesi beni panikletti.", "korku"),
    ("Kapının zorlandığını duyunca elim ayağım titredi.", "korku"),
    ("Sınav sonucumun kötü gelmesinden çok korkuyorum.", "korku"),
    ("Bilinmeyen bir numaradan tehdit mesajı almak beni ürküttü.", "korku"),
    ("Cüzdanımı kaybettiğimi fark edince paniğe kapıldım.", "korku"),
    ("Gece yalnız kalınca içimi büyük bir tedirginlik kaplıyor.", "korku"),
    ("Ani fren yapılınca kaza yapacağız sandım.", "korku"),
    ("Bilgisayarıma virüs bulaştığını düşününce çok endişelendim.", "korku"),
    ("Hesabıma izinsiz giriş yapıldığını görünce korktum.", "korku"),

    ("Bu haberi duyunca ne diyeceğimi bilemedim.", "saskinlik"),
    ("Filmin sonunda katilin hiç beklemediğim biri çıkması beni şaşırttı.", "saskinlik"),
    ("Kapıyı açınca karşımda eski arkadaşımı görünce donakaldım.", "saskinlik"),
    ("Market fiyatlarını görünce gerçekten şaşkına döndüm.", "saskinlik"),
    ("Bir anda herkesin bana sürpriz yapması beni çok şaşırttı.", "saskinlik"),
    ("Sonucun böyle çıkacağını hiç tahmin etmemiştim.", "saskinlik"),
    ("Beklemediğim anda gelen bu haber beni afallattı.", "saskinlik"),
    ("Bu kadar hızlı değişmesine anlam veremedim.", "saskinlik"),
    ("Olayın aslını öğrenince çok şaşırdım.", "saskinlik"),
    ("Bir anda elektriğin kesilmesiyle herkes ne olduğunu anlamadı.", "saskinlik"),
    ("Sınavın iptal edildiğini duyunca çok şaşırdım.", "saskinlik"),
    ("Hiç beklemediğim birinden hediye almak beni şaşırttı.", "saskinlik"),
    ("Karşıma çıkan sonuç beni gerçekten hayrete düşürdü.", "saskinlik"),
    ("O kadar farklı bir cevap verdi ki bir an ne diyeceğimi bilemedim.", "saskinlik"),
    ("Bir anda adımın anons edilmesi beni şaşırttı.", "saskinlik")
]


def build_augmented_dataset():
    base_frame = pd.read_csv(PROCESSED_DATA_PATH)
    extra_frame = pd.DataFrame(EXAMPLES, columns=[TEXT_COLUMN, LABEL_COLUMN])

    if "raw_label" in base_frame.columns:
        base_frame = base_frame[[TEXT_COLUMN, "raw_label", LABEL_COLUMN]]
        extra_frame["raw_label"] = extra_frame[LABEL_COLUMN]
        extra_frame = extra_frame[[TEXT_COLUMN, "raw_label", LABEL_COLUMN]]
    else:
        base_frame = base_frame[[TEXT_COLUMN, LABEL_COLUMN]]

    merged = pd.concat([base_frame, extra_frame], ignore_index=True)
    merged = merged.drop_duplicates(subset=[TEXT_COLUMN, LABEL_COLUMN])
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    output_path = PROCESSED_DATA_PATH.parent / "turkish_tweet_emotion_augmented.csv"
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Genişletilmiş veri seti oluşturuldu.")
    print(f"Dosya: {output_path}")
    print(f"Satır sayısı: {len(merged)}")
    print()
    print(merged[LABEL_COLUMN].value_counts())


if __name__ == "__main__":
    build_augmented_dataset()
