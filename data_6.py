import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. VERİ YÜKLEME VE ÖN İŞLEME (Preprocessing)
zip_path = r'C:\Users\pc\Desktop\Sınavlar\uyz\archive.zip'
print("Aşama 1: Veriler zip dosyasından okunuyor...")

with zipfile.ZipFile(zip_path, 'r') as z:
    thick_file = [f for f in z.namelist() if 'thickness' in f and f.endswith('.csv')][0]
    subj_file = [f for f in z.namelist() if 'subjects' in f and f.endswith('.csv')][0]
    with z.open(thick_file) as f: df_thickness = pd.read_csv(f)
    with z.open(subj_file) as f: df_subjects = pd.read_csv(f)

# Veri Formatlama (Pivot) ve Temizlik
df_thickness['hemi_region'] = df_thickness['hemi'] + '_' + df_thickness['region']
df_pivot = pd.pivot_table(df_thickness, index='subject_id', columns='hemi_region', 
                          values='mean_thickness_weighted', aggfunc='mean').reset_index()

df_merged = pd.merge(df_pivot, df_subjects[['subject_id', 'age']], on='subject_id')
df_merged = df_merged[df_merged['age'] > 0] # Geçersiz yaşları temizle

# Eksik Veri Onarma (Imputation)
numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
df_clean = df_merged.copy()
df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())

# X (Özellikler) ve y (Hedef Yaş) ayırımı
y = df_clean['age']
X = df_clean.drop(columns=['subject_id', 'age'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. MODEL EĞİTİMİ (Optimize Edilmiş Parametreler)
print("Aşama 2: Yapay Zeka modeli eğitiliyor...")
model = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=2, random_state=42)
model.fit(X_train, y_train)
tahminler = model.predict(X_test)

# 3. YAŞ GRUBU REGRESYON ANALİZİ (Hocanın İstediği Tablo)
print("Aşama 3: Yaş gruplarına göre performans analizi yapılıyor...")
sonuc_df = pd.DataFrame({'Gerçek': y_test, 'Tahmin': tahminler, 'Hata': abs(tahminler - y_test)})

bins = [0, 40, 60, 100]
labels = ['Genç (20-40)', 'Orta Yaş (41-60)', 'İleri Yaş (60+)']
sonuc_df['Grup'] = pd.cut(sonuc_df['Gerçek'], bins=bins, labels=labels)

grup_tablosu = sonuc_df.groupby('Grup', observed=False).agg(
    MAE=('Hata', 'mean'),
    Hasta_Sayısı=('Hata', 'count')
).reset_index()

# 4. RAPOR İÇİN METİNSEL ÇIKTILAR (Kopyala-Yapıştır Kısmı)
print("\n" + "="*50)
print("RAPORUN 3.1 (VERİ SETİ) BÖLÜMÜ İÇİN BİLGİ:")
print(f"Toplam Denek Sayısı: {len(df_clean)}")
print(f"Eğitim Seti: {len(X_train)} | Test Seti: {len(X_test)}")
print(f"Öznitelik Sayısı (Beyin Bölgesi): {len(X.columns)}")

print("\nRAPORUN 5 (BULGULAR) BÖLÜMÜ TABLOSU:")
print(grup_tablosu.to_string(index=False))

print("\nPERFORMANS METRİKLERİ:")
print(f"Genel MAE: {mean_absolute_error(y_test, tahminler):.2f} yıl")
print(f"R-Kare (R2): {r2_score(y_test, tahminler):.2f}")
print("="*50)

# 5. GÖRSELLEŞTİRME VE KAYDETME
# Grafik 1: Regresyon Eğilimi
plt.figure(figsize=(10, 6))
plt.scatter(y_test, tahminler, alpha=0.6, color='teal', label='Denekler')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='İdeal Tahmin')
z = np.polyfit(y_test, tahminler, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), "b-", label="Model Eğilimi (Trend)")
plt.title('Chronological Age vs. Predicted Brain Age')
plt.xlabel('Gerçek Yaş')
plt.ylabel('Tahmin Edilen Beyin Yaşı')
plt.legend()
plt.savefig('regresyon_analizi.png', dpi=300)
plt.show()

# Grafik 2: Yaş Grupları Hata Dağılımı
plt.figure(figsize=(9, 6))
bars = plt.bar(grup_tablosu['Grup'], grup_tablosu['MAE'], color=['#4CAF50', '#FFC107', '#F44336'], edgecolor='black')
plt.title('Yaş Gruplarına Göre Ortalama Hata (MAE)')
plt.ylabel('Hata (Yıl)')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.1f} Yıl', ha='center', fontweight='bold')
plt.savefig('yas_grubu_hatalari.png', dpi=300)
plt.show()

# Grafik 3: SHAP Biyobelirteç Analizi
print("Aşama 4: SHAP analizi görselleştiriliyor...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("Beyin Yaşlanması Biyobelirteçleri (SHAP)")
plt.savefig('shap_analizi.png', dpi=300)
plt.show()

print("\nAnaliz tamamlandı. 'regresyon_analizi.png', 'yas_grubu_hatalari.png' ve 'shap_analizi.png' dosyaları kaydedildi.")