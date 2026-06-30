import pandas as pd
import numpy as np
import zipfile
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. VERİ YÜKLEME VE MODEL EĞİTİMİ
zip_path = r'C:\Users\pc\Desktop\Sınavlar\uyz\archive.zip'
print("1. Veriler yükleniyor ve model eğitiliyor...")

with zipfile.ZipFile(zip_path, 'r') as z:
    z_list = z.namelist()
    thick_f = [f for f in z_list if 'thickness' in f and f.endswith('.csv')][0]
    subj_f = [f for f in z_list if 'subjects' in f and f.endswith('.csv')][0]
    with z.open(thick_f) as f: df_thick = pd.read_csv(f)
    with z.open(subj_f) as f: df_subj = pd.read_csv(f)

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

# 2. ANALİZLER BAŞLIYOR
print("2. 5 Farklı Analiz Formatı Oluşturuluyor...")

# --- FORMAT 1: RESIDUAL PLOT (Hata Dağılımı) ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals, color='teal', alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot: Hataların Gerçek Yaşa Göre Dağılımı')
plt.xlabel('Gerçek Yaş')
plt.ylabel('Hata (Tahmin - Gerçek)')
plt.savefig("1_Residual_Plot.pdf", bbox_inches='tight')
plt.show()

# --- FORMAT 2: BLAND-ALTMAN PLOT (Klinik Uyumluluk) ---
mean_age = (y_test + preds) / 2
diff_age = preds - y_test
md = np.mean(diff_age)
sd = np.std(diff_age)

plt.figure(figsize=(10, 6))
plt.scatter(mean_age, diff_age, alpha=0.5, color='darkblue')
plt.axhline(md, color='red', linestyle='-', label=f'Ortalama Fark: {md:.2f}')
plt.axhline(md + 1.96*sd, color='gray', linestyle='--', label='Üst Sınır (95% CI)')
plt.axhline(md - 1.96*sd, color='gray', linestyle='--', label='Alt Sınır (95% CI)')
plt.title('Bland-Altman Plot: YZ Tahmini ve Altın Standart Uyumu')
plt.xlabel('Ortalama Yaş (Gerçek + Tahmin) / 2')
plt.ylabel('Fark (Tahmin - Gerçek)')
plt.legend()
plt.savefig("2_Bland_Altman.pdf", bbox_inches='tight')
plt.show()

# --- FORMAT 3: HATA DAĞILIMI KUTU GRAFİKLERİ (Boxplots) ---
temp_res = pd.DataFrame({'Gerçek': y_test, 'Hata': abs(residuals)})
temp_res['Grup'] = pd.cut(temp_res['Gerçek'], bins=[20, 40, 60, 90], labels=['Genç (20-40)', 'Orta (40-60)', 'İleri (60+)'])

plt.figure(figsize=(10, 6))
sns.boxplot(x='Grup', y='Hata', data=temp_res, palette='Set2')
plt.title('Yaş Gruplarına Göre Hata Dağılımı (MAE)')
plt.ylabel('Mutlak Hata (Yıl)')
plt.savefig("3_Hata_Boxplot.pdf", bbox_inches='tight')
plt.show()

# --- FORMAT 4: SHAP BEESWARM PLOT (Derin Etki Analizi) ---
print("SHAP değerleri hesaplanıyor (Bu işlem biraz vakit alabilir)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('Beyin Bölgelerinin Yaş Tahminine Yönsel Etkisi')
plt.savefig("4_SHAP_Summary_Beeswarm.pdf", bbox_inches='tight')
plt.show()

# --- FORMAT 5: PERFORMANS TABLOSU (Grup Bazlı) ---
def get_metrics(df):
    m = mean_absolute_error(df['Gerçek'], df['Tahmin'])
    r = r2_score(df['Gerçek'], df['Tahmin'])
    return pd.Series([len(df), m, r], index=['Denek Sayısı', 'MAE', 'R2'])

perf_df = pd.DataFrame({'Gerçek': y_test, 'Tahmin': preds})
perf_df['Grup'] = pd.cut(perf_df['Gerçek'], bins=[20, 40, 60, 90], labels=['Genç', 'Orta', 'İleri'])
tablo = perf_df.groupby('Grup').apply(get_metrics)

print("\n--- PERFORMANS ÖZET TABLOSU ---")
print(tablo.to_string())
tablo.to_excel("Performans_Ozeti_Tablo.xlsx")