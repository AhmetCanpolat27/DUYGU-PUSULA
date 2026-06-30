import pandas as pd

path = "outputs/metrics/manual_test_predictions.csv"
frame = pd.read_csv(path)

v2_wrong = frame[(frame["model"] == "berturk_v2") & (frame["correct"] == False)].copy()
v2_wrong = v2_wrong.sort_values(by=["actual_label", "predicted_label", "confidence"], ascending=[True, True, False])

print("BERTurk v2 hatalı tahmin sayısı:", len(v2_wrong))
print()

print("Hata dağılımı:")
print(v2_wrong.groupby(["actual_label", "predicted_label"]).size().reset_index(name="count"))
print()

print("Hatalı örnekler:")
for _, row in v2_wrong.iterrows():
    print("-" * 80)
    print("Metin:", row["text"])
    print("Gerçek:", row["actual_label"])
    print("Tahmin:", row["predicted_label"])
    print(f"Güven: %{row['confidence'] * 100:.2f}")

v2_wrong.to_csv("outputs/metrics/berturk_v2_wrong_predictions_only.csv", index=False, encoding="utf-8-sig")
print()
print("Dosya kaydedildi: outputs/metrics/berturk_v2_wrong_predictions_only.csv")
