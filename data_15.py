import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. VERİ YÜKLEME VE ÖN HAZIRLIK
zip_path = r'C:\Users\pc\Desktop\Sınavlar\uyz\archive.zip'
print("1. Veriler yükleniyor ve cinsiyet analizi hazırlanıyor...")

with zipfile.ZipFile(zip_path, 'r') as z:
    z_list = z.namelist()
    thick_f = [f for f in z_list if 'thickness' in f and f.endswith('.csv')][0]
    subj_f = [f for f in z_list if 'subjects' in f and f.endswith('.csv')][0]
    with z.open(thick_f) as f: df_thick = pd.read_csv(f)
    with z.open(subj_f) as f: df_subj = pd.read_csv(f)

# Ön işleme ve Birleştirme
df_thick['hemi_region'] = df_thick['hemi'] + '_' + df_thick['region']
df_piv = pd.pivot_table(df_thick, index='subject_id', columns='hemi_region', values='mean_thickness_weighted', aggfunc='mean').reset_index()

# Cinsiyet (sex) bilgisini de içeren birleştirme
df_m = pd.merge(df_piv, df_subj[['subject_id', 'age', 'sex']], on='subject_id')
df_m = df_m[df_m['age'] > 0].fillna(df_m.mean(numeric_only=True))

# Eğitim
X = df_m[df_piv.columns].drop(columns=['subject_id'])
y = df_m['age']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("2. Yapay Zeka eğitiliyor...")
model = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_train, y_train)
preds = model.predict(X_test)

# 2. CİNSİYET BAZLI HATA HESAPLAMA
res = df_m.loc[X_test.index].copy()
res['Tahmin'] = preds
res['Hata'] = abs(res['Tahmin'] - res['age'])

# Cinsiyet kodlarını (1/2 veya M/F) anlamlı hale getirelim
# IXI veri setinde genellikle 1=Erkek, 2=Kadın'dır (veya tam tersi, terminal çıktısından kontrol edebilirsin)
cinsiyet_analizi = res.groupby('sex')['Hata'].mean().reset_index()
cinsiyet_analizi.columns = ['Cinsiyet', 'MAE']

# 3. GÖRSELLEŞTİRME
plt.figure(figsize=(8, 6))
colors = ['skyblue', 'salmon']
bars = plt.bar(cinsiyet_analizi['Cinsiyet'].astype(str), cinsiyet_analizi['MAE'], color=colors, edgecolor='black')

plt.title('Cinsiyete Göre Model Hata Payı Karşılaştırması', fontsize=14)
plt.xlabel('Cinsiyet (1: Erkek / 2: Kadın)', fontsize=12)
plt.ylabel('Ortalama Hata (MAE - Yıl)', fontsize=12)

# Değerleri barların üzerine yazdır
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f} Yıl', ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# PDF Kaydetme
plt.savefig("cinsiyet_hata_analizi.pdf", format='pdf', bbox_inches='tight')
print("Grafik 'cinsiyet_hata_analizi.pdf' adıyla kaydedildi.")

plt.show()

# Terminale özet basalım
print("\n--- CİNSİYET BAZLI PERFORMANS ---")
print(cinsiyet_analizi.to_string(index=False))