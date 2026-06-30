import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. VERİ YÜKLEME VE HAZIRLIK
zip_path = r'C:\Users\pc\Desktop\Sınavlar\uyz\archive.zip'
print("1. 579 hastanın verileri titizlikle işleniyor...")

with zipfile.ZipFile(zip_path, 'r') as z:
    z_list = z.namelist()
    thick_f = [f for f in z_list if 'thickness' in f and f.endswith('.csv')][0]
    subj_f = [f for f in z_list if 'subjects' in f and f.endswith('.csv')][0]
    with z.open(thick_f) as f: df_thick = pd.read_csv(f)
    with z.open(subj_f) as f: df_subj = pd.read_csv(f)

df_thick['hemi_region'] = df_thick['hemi'] + '_' + df_thick['region']
df_piv = pd.pivot_table(df_thick, index='subject_id', columns='hemi_region', values='mean_thickness_weighted', aggfunc='mean').reset_index()

# Verileri birleştirme
df_all = pd.merge(df_piv, df_subj, on='subject_id')
df_all = df_all[df_all['age'] > 0].fillna(df_all.mean(numeric_only=True))

# 2. MODEL EĞİTİMİ VE TAHMİN
X = df_all[df_piv.columns].drop(columns=['subject_id'])
y = df_all['age']

# Test/Train ayırarak modeli eğitme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_train, y_train)

# Tüm 579 hasta için tahmin üret
df_all['Tahmin_Edilen_Yas'] = model.predict(X)
df_all['Brain_Age_Gap'] = df_all['Tahmin_Edilen_Yas'] - df_all['age']
df_all['Mutlak_Hata'] = abs(df_all['Brain_Age_Gap'])

# 3. EXCEL KAYDI (.xlsx formatı)
excel_adi = "579_Hasta_Detayli_Analiz_Raporu.xlsx"

export_cols = ['subject_id', 'age', 'Tahmin_Edilen_Yas', 'Brain_Age_Gap', 'Mutlak_Hata', 'sex', 'scanner', 'site']
df_all[export_cols].to_excel(excel_adi, index=False)
print(f"-> Excel dosyası '{excel_adi}' adıyla kusursuz şekilde oluşturuldu.")

# 4. GÖRSELLEŞTİRME 
plt.figure(figsize=(12, 8))
sns.regplot(x='age', y='Tahmin_Edilen_Yas', data=df_all, 
            scatter_kws={'alpha':0.4, 'color':'darkblue'}, 
            line_kws={'color':'crimson', 'label':'Regresyon Doğrusu'})

plt.plot([y.min(), y.max()], [y.min(), y.max()], 'g--', alpha=0.6, label='İdeal Tahmin (Gap=0)')

mae = mean_absolute_error(df_all['age'], df_all['Tahmin_Edilen_Yas'])
r2 = r2_score(df_all['age'], df_all['Tahmin_Edilen_Yas'])

plt.text(25, 75, f"Denek Sayısı: {len(df_all)}\nGenel MAE: {mae:.2f} Yıl\nR² Skoru: {r2:.2f}", 
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'), fontsize=12)

plt.title('Tüm Popülasyon Tahmin Analizi (N=579)', fontsize=16, fontweight='bold')
plt.xlabel('Gerçek Yaş', fontsize=12)
plt.ylabel('Yapay Zeka Tahmini', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("579_Hasta_Genel_Grafik.pdf", format='pdf', bbox_inches='tight')
plt.show()

print(f"\nİşlem Tamam! Toplam {len(df_all)} hasta için veriler Excel'e aktarıldı.")