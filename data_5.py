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

# -1.0 yaş skandalını temizliyoruz
df_merged = df_merged[df_merged['age'] > 0]

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

# 3. YAŞ GRUPLARI ANALİZİ (HOCANIN İSTEDİĞİ TABLO)
sonuc_df = pd.DataFrame({
    'Gercek_Yas': y_test,
    'Tahmini_Yas': beyin_yaslari,
    'Hata_Payi': abs(beyin_yaslari - y_test)
})

# Grupları biraz daha mantıklı bölelim ki boş kalmasın
bins = [0, 40, 60, 100]
labels = ['Genç (20-40 Yaş)', 'Orta Yaş (41-60 Yaş)', 'İleri Yaş (60+ Yaş)']
sonuc_df['Yas_Grubu'] = pd.cut(sonuc_df['Gercek_Yas'], bins=bins, labels=labels)

# Her grupta kaç hasta var ve ortalama hata ne kadar?
grup_analizi = sonuc_df.groupby('Yas_Grubu', observed=False).agg(
    Ortalama_Hata=('Hata_Payi', 'mean'),
    Hasta_Sayisi=('Hata_Payi', 'count')
).reset_index()

print("--- RAPOR İÇİN YAŞ GRUPLARI ANALİZİ ---")
print(grup_analizi.to_string(index=False))

# =========================================================
# 4. GÖRSELLEŞTİRME 1: ANLAŞILIR SÜTUN GRAFİĞİ
# =========================================================
plt.figure(figsize=(9, 6))
renkler = ['#4CAF50', '#FFC107', '#F44336'] # Yeşil, Sarı, Kırmızı
çubuklar = plt.bar(grup_analizi['Yas_Grubu'], grup_analizi['Ortalama_Hata'], color=renkler, edgecolor='black')

# Barların tepesine net değerleri ve hasta sayılarını yazdırıyoruz (Veri az karışıklığını çözer)
for bar, count in zip(çubuklar, grup_analizi['Hasta_Sayisi']):
    yval = bar.get_height()
    if pd.notna(yval):
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.3, 
                 f'Hata: {yval:.1f} Yıl\n(N={count} Kişi)', 
                 ha='center', va='bottom', fontweight='bold')

plt.title('Yaş Gruplarına Göre Yapay Zekanın Ortalama Hata Payı (MAE)', fontsize=14)
plt.xlabel('Yaş Grupları', fontsize=12)
plt.ylabel('Ortalama Hata Payı (Yıl)', fontsize=12)
plt.ylim(0, grup_analizi['Ortalama_Hata'].max() + 3) # Tepede yazı için boşluk bırak
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# =========================================================
# 5. GÖRSELLEŞTİRME 2: GENÇLERİ YAŞLI SANMA KANITI (TREND ÇİZGİSİ)
# =========================================================
plt.figure(figsize=(9, 6))
plt.scatter(y_test, beyin_yaslari, alpha=0.7, color='teal', edgecolors='white', s=60, label='Hastalar')

# Kusursuz tahmin çizgisi (Kırmızı kesikli)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Kusursuz Tahmin (0 Hata)')

# Yapay Zekanın GERÇEK EĞİLİM ÇİZGİSİ (Mavi düz çizgi)
z = np.polyfit(y_test, beyin_yaslari, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), "b-", lw=3, label="Yapay Zekanın Eğilimi (Trend)")

plt.title('Ortalamaya Regresyon: Gençleri Yaşlı, Yaşlıları Genç Sanma Eğilimi', fontsize=14)
plt.xlabel('Gerçek Kimlik Yaşı', fontsize=12)
plt.ylabel('YZ Tahmini (Brain Age)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()