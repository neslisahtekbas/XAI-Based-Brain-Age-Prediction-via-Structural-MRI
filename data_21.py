import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Model Kütüphaneleri
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# 1. VERİ HAZIRLIĞI
zip_path = r'C:\Users\pc\Desktop\Sınavlar\uyz\archive.zip'
print("1. Veriler yükleniyor...")

with zipfile.ZipFile(zip_path, 'r') as z:
    z_list = z.namelist()
    thick_f = [f for f in z_list if 'thickness' in f and f.endswith('.csv')][0]
    subj_f = [f for f in z_list if 'subjects' in f and f.endswith('.csv')][0]
    with z.open(thick_f) as f: df_thick = pd.read_csv(f)
    with z.open(subj_f) as f: df_subj = pd.read_csv(f)

df_thick['hemi_region'] = df_thick['hemi'] + '_' + df_thick['region']
df_piv = pd.pivot_table(df_thick, index='subject_id', columns='hemi_region', values='mean_thickness_weighted', aggfunc='mean').reset_index()
df_m = pd.merge(df_piv, df_subj[['subject_id', 'age']], on='subject_id')
df_m = df_m[df_m['age'] > 0].fillna(df_m.mean(numeric_only=True))

X = df_m[df_piv.columns].drop(columns=['subject_id'])
y = df_m['age']

# MLP ve SVR gibi modeller ölçeklendirilmiş veri ister
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. MODELLERİ TANIMLAMA
modeller = {
    "Lineer (Ridge)": Ridge(),
    "SVR (Vektör Destek)": SVR(kernel='rbf'),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
    "Yapay Sinir Ağları (MLP)": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

sonuclar = []

print("2. Modeller eğitiliyor ve kapıştırılıyor (Lütfen bekleyin)...")
for isim, model in modeller.items():
    model.fit(X_train, y_train)
    tahmin = model.predict(X_test)
    mae = mean_absolute_error(y_test, tahmin)
    r2 = r2_score(y_test, tahmin)
    sonuclar.append({"Model": isim, "MAE": mae, "R2": r2})
    print(f"-> {isim} bitti.")

df_sonuc = pd.DataFrame(sonuclar).sort_values("MAE")

# 3. GÖRSELLEŞTİRME (KARŞILAŞTIRMA)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# MAE Grafiği (Düşük olan iyidir)
sns.barplot(x='MAE', y='Model', data=df_sonuc, palette='viridis', ax=ax1)
ax1.set_title('Model Hata Payları (MAE - Düşük İyidir)', fontweight='bold')
ax1.set_xlabel('Ortalama Mutlak Hata (Yıl)')

# R2 Grafiği (Yüksek olan iyidir)
sns.barplot(x='R2', y='Model', data=df_sonuc.sort_values("R2", ascending=False), palette='magma', ax=ax2)
ax2.set_title('Model Başarı Skorları (R2 - Yüksek İyidir)', fontweight='bold')
ax2.set_xlabel('R-Kare Skoru')

plt.tight_layout()
plt.savefig("Model_Karsilastirma_Analizi.pdf", bbox_inches='tight')
plt.show()

# 4. FİNAL HATA BANDI GRAFİĞİ (En İyi Model İçin)
# Genelde Random Forest veya Boosting en iyisi çıkar
en_iyi_model_ismi = df_sonuc.iloc[0]['Model']
en_iyi_mae = df_sonuc.iloc[0]['MAE']

plt.figure(figsize=(10, 10))
en_iyi_model = modeller[en_iyi_model_ismi]
final_tahmin = en_iyi_model.predict(X_test)

plt.scatter(y_test, final_tahmin, alpha=0.5, color='navy', label='Denekler')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='İdeal Çizgi (0 Hata)')
plt.fill_between([y.min(), y.max()], [y.min()-en_iyi_mae, y.max()-en_iyi_mae], 
                 [y.min()+en_iyi_mae, y.max()+en_iyi_mae], color='gray', alpha=0.2, label=f'Hata Bandı (±{en_iyi_mae:.2f} Yıl)')

plt.title(f'FİNAL ANALİZİ: {en_iyi_model_ismi} Performansı', fontsize=15, fontweight='bold')
plt.xlabel('Gerçek Kronolojik Yaş')
plt.ylabel('YZ Tahmini Beyin Yaşı')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig("FINAL_HATA_BANDI.pdf", bbox_inches='tight')
plt.show()

# 5. SONUÇLARI EXCEL'E YAZ
df_sonuc.to_excel("Model_Karsilastirma_Tablosu.xlsx", index=False)
print("\n--- MODEL KIYASLAMA TABLOSU ---")
print(df_sonuc.to_string(index=False))