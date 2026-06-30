import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. VERİ YÜKLEME VE MODEL HAZIRLIĞI
zip_path = r'C:\Users\pc\Desktop\Sınavlar\uyz\archive.zip'
print("Analizler hazırlanıyor...")

with zipfile.ZipFile(zip_path, 'r') as z:
    z_list = z.namelist()
    thick_f = [f for f in z_list if 'thickness' in f and f.endswith('.csv')][0]
    subj_f = [f for f in z_list if 'subjects' in f and f.endswith('.csv')][0]
    with z.open(thick_f) as f: df_thick = pd.read_csv(f)
    with z.open(subj_f) as f: df_subj = pd.read_csv(f)

# Ön işleme
df_thick['hemi_region'] = df_thick['hemi'] + '_' + df_thick['region']
df_piv = pd.pivot_table(df_thick, index='subject_id', columns='hemi_region', values='mean_thickness_weighted', aggfunc='mean').reset_index()
df_all = pd.merge(df_piv, df_subj, on='subject_id')
df_all = df_all[df_all['age'] > 0].fillna(df_all.mean(numeric_only=True))

X = df_all[df_piv.columns].drop(columns=['subject_id'])
y = df_all['age']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_train, y_train)
preds = model.predict(X_test)
residuals = preds - y_test

# --- 1. RESIDUAL PLOT (Hataların Yaşa Göre Dağılımı) ---
plt.figure(figsize=(8, 5))
plt.scatter(y_test, residuals, color='teal', alpha=0.6, edgecolors='k')
plt.axhline(0, color='red', linestyle='--')
plt.title('1. Residual Plot: Hata Dağılım Analizi')
plt.xlabel('Gerçek Yaş'); plt.ylabel('Hata (Tahmin - Gerçek)')
plt.savefig("analiz_1_residual.pdf", bbox_inches='tight')

# --- 2. BLAND-ALTMAN PLOT (Metot Uyumluluğu) ---
mean_val = (y_test + preds) / 2
plt.figure(figsize=(8, 5))
plt.scatter(mean_val, residuals, alpha=0.5, color='darkblue')
plt.axhline(np.mean(residuals), color='red', label='Ortalama Fark')
plt.axhline(np.mean(residuals) + 1.96*np.std(residuals), color='gray', linestyle='--', label='95% Güven Sınırı')
plt.axhline(np.mean(residuals) - 1.96*np.std(residuals), color='gray', linestyle='--')
plt.title('2. Bland-Altman Analizi: Klinik Uyumluluk')
plt.xlabel('Ortalama Yaş'); plt.ylabel('Fark (Tahmin - Gerçek)')
plt.legend(); plt.savefig("analiz_2_bland_altman.pdf", bbox_inches='tight')

# --- 3. HATA BOXPLOT (Grup Bazlı Sapmalar) ---
temp_res = pd.DataFrame({'Gerçek': y_test, 'Hata': abs(residuals)})
temp_res['Grup'] = pd.cut(temp_res['Gerçek'], bins=[20, 40, 60, 90], labels=['20-40', '40-60', '60+'])
plt.figure(figsize=(8, 5))
sns.boxplot(x='Grup', y='Hata', data=temp_res, palette='Set2')
plt.title('3. Yaş Gruplarına Göre MAE Dağılımı'); plt.ylabel('Mutlak Hata (Yıl)')
plt.savefig("analiz_3_boxplot.pdf", bbox_inches='tight')

# --- 4. FEATURE IMPORTANCE (Önemli Bölgeler) ---
feats = pd.Series(model.feature_importances_, index=X.columns).sort_values().tail(15)
plt.figure(figsize=(10, 6))
feats.plot(kind='barh', color='orchid')
plt.title('4. Karar Sürecinde En Etkili 15 Beyin Bölgesi')
plt.savefig("analiz_4_importance.pdf", bbox_inches='tight')

# --- 5. PERFORMANS TABLOSU (Görsel Olarak) ---
perf_data = [['Genel MAE', f"{mean_absolute_error(y_test, preds):.2f} Yıl"],
             ['R2 Skoru', f"{r2_score(y_test, preds):.2f}"],
             ['En Başarılı Grup', 'Genç (20-40)']]
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('tight'); ax.axis('off')
ax.table(cellText=perf_data, colLabels=['Metrik', 'Değer'], loc='center')
plt.savefig("analiz_5_tablo.pdf", bbox_inches='tight')

# --- 6. FINAL GRAFİK: ERROR BANDS (Hata Bantlı Regresyon) ---
plt.figure(figsize=(9, 9))
plt.scatter(y_test, preds, alpha=0.4, color='navy', label='Denekler')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='İdeal Tahmin (Gap=0)')
# Hata payı bandını (MAE) ekleyelim
mae = mean_absolute_error(y_test, preds)
plt.fill_between([y.min(), y.max()], [y.min()-mae, y.max()-mae], [y.min()+mae, y.max()+mae], 
                 color='gray', alpha=0.15, label=f'Model Güven Aralığı (±{mae:.2f} Yıl)')
plt.title('FİNAL: Gerçek Yaş vs. Beyin Yaşı Tahmini', fontsize=15, fontweight='bold')
plt.xlabel('Gerçek Kronolojik Yaş'); plt.ylabel('YZ Tahmini Beyin Yaşı')
plt.legend(); plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig("analiz_6_FINAL_GRAFIK.pdf", bbox_inches='tight')

plt.show()
print("Tüm grafikler (1-6) ayrı PDF'ler olarak kaydedildi!")