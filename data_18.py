import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# 1. VERİ YÜKLEME
zip_path = r'C:\Users\pc\Desktop\Sınavlar\uyz\archive.zip'
print("1. Korelasyon analizi için veriler harmanlanıyor...")

with zipfile.ZipFile(zip_path, 'r') as z:
    z_list = z.namelist()
    thick_f = [f for f in z_list if 'thickness' in f and f.endswith('.csv')][0]
    subj_f = [f for f in z_list if 'subjects' in f and f.endswith('.csv')][0]
    with z.open(thick_f) as f: df_thick = pd.read_csv(f)
    with z.open(subj_f) as f: df_subj = pd.read_csv(f)

# Ön işleme
df_thick['hemi_region'] = df_thick['hemi'] + '_' + df_thick['region']
df_piv = pd.pivot_table(df_thick, index='subject_id', columns='hemi_region', values='mean_thickness_weighted', aggfunc='mean').reset_index()
df_all = pd.merge(df_piv, df_subj[['subject_id', 'age']], on='subject_id')
df_all = df_all[df_all['age'] > 0].fillna(df_all.mean(numeric_only=True))

# 2. EN ÖNEMLİ 15 BÖLGEYİ SEÇME 
numeric_df = df_all.drop(columns=['subject_id'])
correlations = numeric_df.corr()['age'].abs().sort_values(ascending=False)
top_15_regions = correlations.index[:16] 

# Sadece bu bölgelerin korelasyon matrisini çıkar
reduced_corr_matrix = numeric_df[top_15_regions].corr()

# 3. GÖRSELLEŞTİRME (HEATMAP)
plt.figure(figsize=(14, 10))
sns.heatmap(reduced_corr_matrix, annot=True, cmap='RdBu_r', fmt=".2f", linewidths=0.5)

plt.title('Yaşlanma ile En İlişkili Beyin Bölgeleri: Korelasyon Isı Haritası', fontsize=16, fontweight='bold')
plt.tight_layout()


plt.savefig("beyin_bolgeleri_korelasyon_heatmap.pdf", format='pdf', bbox_inches='tight')
print("Isı haritası 'beyin_bolgeleri_korelasyon_heatmap.pdf' adıyla kaydedildi.")

plt.show()

print("\n--- YAŞLA EN GÜÇLÜ NEGATİF KORELASYONU OLAN 5 BÖLGE ---")
print("(Bu bölgeler yaşlandıkça en çok incelen yerlerdir)")
print(correlations[1:6])