import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. VERİ YÜKLEME VE DÜZENLEME (Artık burası standart)
zip_path = r'C:\Users\pc\Desktop\Sınavlar\uyz\archive.zip'
print("1. Veriler okunuyor ve hazırlanıyor...")

with zipfile.ZipFile(zip_path, 'r') as z:
    thick_file = [f for f in z.namelist() if 'thickness' in f and f.endswith('.csv')][0]
    subj_file = [f for f in z.namelist() if 'subjects' in f and f.endswith('.csv')][0]
    with z.open(thick_file) as f: df_thickness = pd.read_csv(f)
    with z.open(subj_file) as f: df_subjects = pd.read_csv(f)

df_thickness['hemi_region'] = df_thickness['hemi'] + '_' + df_thickness['region']
df_pivot = pd.pivot_table(df_thickness, index='subject_id', columns='hemi_region', values='mean_thickness_weighted', aggfunc='mean').reset_index()
df_merged = pd.merge(df_pivot, df_subjects[['subject_id', 'age']], on='subject_id')

sayisal_sutunlar = df_merged.select_dtypes(include=['float64', 'int64']).columns
df_clean = df_merged.copy()
df_clean[sayisal_sutunlar] = df_clean[sayisal_sutunlar].fillna(df_clean[sayisal_sutunlar].mean())

y = df_clean['age'] 
X = df_clean.drop(columns=['subject_id', 'age']) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. MODELİ EĞİTME
print("2. Yapay Zeka eğitiliyor...")
model = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=2, random_state=42)
model.fit(X_train, y_train)
beyin_yaslari = model.predict(X_test)

# =========================================================
# 3. RAPOR İÇİN İSTATİSTİKSEL METRİKLER (Burayı Raporun 5. Bölümüne Yaz)
# =========================================================
mae = mean_absolute_error(y_test, beyin_yaslari)
r2 = r2_score(y_test, beyin_yaslari)
korelasyon = np.corrcoef(y_test, beyin_yaslari)[0, 1]

print("\n--- RAPOR TABLO 2 İÇİN VERİLER ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} yıl")
print(f"R-Kare (R²) Skoru: {r2:.2f} (Literatürdeki standartlara uygun muhtemelen 0.30 - 0.50 arası çıkacaktır)")
print(f"Pearson Korelasyonu (r): {korelasyon:.2f} (Modelin gerçek yaşla ne kadar paralel ilerlediğini gösterir)")

# =========================================================
# 4. BRAIN AGE GAP (BEYİN YAŞI FARKI) VAKA ANALİZLERİ
# =========================================================
# Gap = Tahmini Yaş - Gerçek Yaş
gap_df = pd.DataFrame({
    'Gercek_Yas': y_test,
    'Tahmini_Yas': beyin_yaslari,
    'Brain_Age_Gap': beyin_yaslari - y_test
})

# En dramatik örnekleri seçelim
erken_yaslananlar = gap_df.sort_values(by='Brain_Age_Gap', ascending=False).head(3)
genc_kalanlar = gap_df.sort_values(by='Brain_Age_Gap', ascending=True).head(3)

print("\n--- RAPOR 'VAKA ÇALIŞMASI' İÇİN EN UÇ ÖRNEKLER ---")
print("Hızlanmış Yaşlanma Gösteren 3 Hasta (Tahmini > Gerçek):")
print(erken_yaslananlar.to_string())

print("\nBeyni Genç Kalan (Super-Ager) 3 Hasta (Tahmini < Gerçek):")
print(genc_kalanlar.to_string())

# =========================================================
# 5. RAPOR ŞEKİL 1 İÇİN: "RESIDUAL PLOT" (HATA DAĞILIMI GRAFİĞİ)
# =========================================================
print("\n-> Rapor için Hata Dağılım (Gap) Grafiği Çiziliyor...")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, gap_df['Brain_Age_Gap'], alpha=0.7, color='coral', edgecolors='black', s=50)

# Kusursuz sıfır hatası çizgisi
plt.axhline(y=0, color='red', linestyle='--', lw=2, label='Sıfır Hata (Gerçek = Tahmin)')

plt.title('Brain Age Gap Dağılımı (Yaşa Göre Tahmin Sapmaları)', fontsize=14)
plt.xlabel('Gerçek Kimlik Yaşı (Chronological Age)', fontsize=12)
plt.ylabel('Beyin Yaşı Farkı (Gap) = Tahmin - Gerçek', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Not: Eğer SHAP grafiğini de tekrar görmek istersen kodu buraya ekleyebilirsin.