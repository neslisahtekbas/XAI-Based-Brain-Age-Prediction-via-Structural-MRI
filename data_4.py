import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. VERİ YÜKLEME VE DÜZENLEME
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

# =========================================================
# KRİTİK DÜZELTME: Yaşı -1 olan hatalı hastaları siliyoruz!
df_merged = df_merged[df_merged['age'] > 0]
# =========================================================

sayisal_sutunlar = df_merged.select_dtypes(include=['float64', 'int64']).columns
df_clean = df_merged.copy()
df_clean[sayisal_sutunlar] = df_clean[sayisal_sutunlar].fillna(df_clean[sayisal_sutunlar].mean())

y = df_clean['age'] 
X = df_clean.drop(columns=['subject_id', 'age']) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. MODELİ EĞİTME
print("2. Yapay Zeka eğitiliyor...\n")
model = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=2, random_state=42)
model.fit(X_train, y_train)
beyin_yaslari = model.predict(X_test)

# 3. GENEL İSTATİSTİKLER
mae = mean_absolute_error(y_test, beyin_yaslari)
r2 = r2_score(y_test, beyin_yaslari)

# =========================================================
# 4. HOCANIN İSTEDİĞİ: YAŞ GRUPLARINA GÖRE ANALİZ
# =========================================================
# Sonuçları tek bir tabloda toplayalım
sonuc_df = pd.DataFrame({
    'Gercek_Yas': y_test,
    'Tahmini_Yas': beyin_yaslari,
    'Hata_Payi': abs(beyin_yaslari - y_test) # Mutlak hata
})

# Hastaları 3 gruba ayırıyoruz: Genç (<45), Orta Yaş (45-60), İleri Yaş (>60)
bins = [0, 45, 60, 100]
labels = ['Genç (<45)', 'Orta Yaş (45-60)', 'İleri Yaş (>60)']
sonuc_df['Yas_Grubu'] = pd.cut(sonuc_df['Gercek_Yas'], bins=bins, labels=labels)

# Her grup için ortalama hata payını (MAE) hesaplıyoruz
grup_analizi = sonuc_df.groupby('Yas_Grubu')['Hata_Payi'].mean().reset_index()
grup_analizi.columns = ['Yaş Grubu', 'Ortalama Sapma (Yıl)']

print("--- RAPOR İÇİN YAŞ GRUPLARI ANALİZİ (Hocanın İstediği Tablo) ---")
print(grup_analizi.to_string(index=False))
print(f"\nGenel Ortalama Hata (Tüm Gruplar): {mae:.2f} Yıl")
print(f"Genel R-Kare (R²): {r2:.2f}")

# 5. GÖRSELLEŞTİRME: KUTU GRAFİĞİ (Boxplot) - Rapor İçin Çok Uygun
plt.figure(figsize=(8, 6))
sonuc_df.boxplot(column='Hata_Payi', by='Yas_Grubu', grid=False, color='blue')
plt.title('Yaş Gruplarına Göre Yapay Zekanın Hata Payı Dağılımı')
plt.suptitle('') # Otomatik gelen gereksiz başlığı siler
plt.xlabel('Yaş Grupları')
plt.ylabel('Hata Payı (Yıl)')
plt.show()