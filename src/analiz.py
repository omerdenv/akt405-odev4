import yfinance as yf
import pandas as pd
import numpy as np
import os

# Varlıklar
varliklar = {
    "THYAO": "THYAO.IS",
    "ASELS": "ASELS.IS",
    "BIST100": "XU100.IS"
}

# Tarih aralığı
baslangic = "2021-01-01"
bitis = "2025-12-31"

# Klasör oluştur
os.makedirs("data", exist_ok=True)

# Kapanış fiyatlarını tutacak tablo
kapanis = pd.DataFrame()

# Veri çekme
for isim, kod in varliklar.items():
    print(isim, "verisi indiriliyor...")

    veri = yf.download(
        kod,
        start=baslangic,
        end=bitis,
        auto_adjust=False
    )

    kapanis[isim] = veri["Close"][kod]

# Eksik verileri temizle
kapanis = kapanis.dropna()

# Kapanış fiyatlarını kaydet
kapanis.to_csv("data/kapanis.csv")


# -----------------------------
# GETİRİ VE RİSK HESAPLAMA
# -----------------------------

# Günlük basit getiriler
getiriler = kapanis.pct_change().dropna()

# Log getiriler
log_getiriler = np.log(kapanis / kapanis.shift(1))
log_getiriler = log_getiriler.dropna()

# Yıllık ortalama getiri
yillik_getiri = getiriler.mean() * 252

# Yıllık risk (volatilite)
yillik_risk = getiriler.std() * np.sqrt(252)

# Sonuç tablosu
sonuc = pd.DataFrame({
    "Yillik Getiri": yillik_getiri,
    "Yillik Risk": yillik_risk
})


print("\nGetiri ve Risk Tablosu:")
print(sonuc)


# Sonucu kaydet
sonuc.to_csv("data/getiri_risk.csv")

print("\nTum islemler tamamlandi.")
# -----------------------------
# GRAFİKLER
# -----------------------------

import matplotlib.pyplot as plt

# Portföy performansı (100 TL başlangıç)
performans = (1 + getiriler).cumprod() * 100

plt.figure(figsize=(10, 5))

for kolon in performans.columns:
    plt.plot(performans.index, performans[kolon], label=kolon)

plt.title("100 TL Baslangic ile Varlik Performansi")
plt.xlabel("Tarih")
plt.ylabel("Portfoy Degeri (TL)")
plt.legend()
plt.grid()

plt.savefig("plots/portfoy.png", bbox_inches="tight")
plt.close()


# Korelasyon matrisi

korelasyon = getiriler.corr()

plt.figure(figsize=(7, 5))

plt.imshow(korelasyon)

plt.colorbar()

plt.xticks(
    range(len(korelasyon.columns)),
    korelasyon.columns
)

plt.yticks(
    range(len(korelasyon.columns)),
    korelasyon.columns
)

plt.title("Getiri Korelasyon Matrisi")

for i in range(len(korelasyon)):
    for j in range(len(korelasyon)):
        plt.text(
            j,
            i,
            round(korelasyon.iloc[i, j], 2),
            ha="center",
            va="center"
        )

plt.savefig("plots/korelasyon.png", bbox_inches="tight")
plt.close()


print("Grafikler oluşturuldu.")