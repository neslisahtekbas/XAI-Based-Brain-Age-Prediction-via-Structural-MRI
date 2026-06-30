import pandas as pd
import numpy as np
import zipfile
import nibabel as nib
import matplotlib.pyplot as plt
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. VERİ YÜKLEME VE MODEL EĞİTİMİ
zip_path = r'C:\Users\pc\Desktop\Sınavlar\uyz\archive.zip'
print("1. Veriler yükleniyor...")

with zipfile.ZipFile(zip_path, 'r') as z:
    thick_f = [f for f in z.namelist() if 'thickness' in f and f.endswith('.csv')][0]
    subj_f = [f for f in z.namelist() if 'subjects' in f and f.endswith('.csv')][0]
    with z.open(thick_f) as f: df_thick = pd.read_csv(f)
    with z.open(subj_f) as f: df_subj = pd.read_csv(f)

# Ön işleme
df_thick['hemi_region'] = df_thick['hemi'] + '_' + df_thick['region']
df_piv = pd.pivot_table(df_thick, index='subject_id', columns='hemi_region', values='mean_thickness_weighted', aggfunc='mean').reset_index()
df_m = pd.merge(df_piv, df_subj, on='subject_id')
df_m = df_m[df_m['age'] > 0].fillna(df_m.mean(numeric_only=True))

# Eğitim
X = df_m[df_piv.columns].drop(columns=['subject_id'])
y = df_m['age']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("2. Yapay Zeka eğitiliyor...")
model = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_train, y_train)
preds = model.predict(X_test)

# 2. EN HATALI 5 HASTAYI (OUTLIERS) SEÇME
test_indices = X_test.index
res = df_m.loc[test_indices].copy()
res['Tahmin'] = preds
res['Hata'] = abs(res['Tahmin'] - res['age'])

# Hata payına göre (büyükten küçüğe) sıralayıp ilk 5'i alıyoruz
worst_cases = res.sort_values('Hata', ascending=False).head(5)

# 3. 5 HASTA X 3 DÜZLEM (15 FOTOĞRAF) GÖRSELLEŞTİRME
print("3. En yüksek hatalı 5 hastanın MR kesitleri hazırlanıyor...")
with zipfile.ZipFile(zip_path, 'r') as z:
    z_list = z.namelist()
    fig, axes = plt.subplots(5, 3, figsize=(15, 25)) # 5 Satır, 3 Sütun
    planes = ["Sagittal (X)", "Coronal (Y)", "Axial (Z)"]

    for i, (idx, row) in enumerate(worst_cases.iterrows()):
        # Dosyayı zip içinde bul
        found = [f for f in z_list if row['subject_id'] in f and f.endswith('.nii')]
        
        if found:
            with z.open(found[0]) as f:
                content = io.BytesIO(f.read())
                img = nib.Nifti1Image.from_file_map({'header': nib.FileHolder(fileobj=content), 
                                                      'image': nib.FileHolder(fileobj=content)})
                data = img.get_fdata()
                mx, my, mz = data.shape[0]//2, data.shape[1]//2, data.shape[2]//2
                
                # 3 Düzlem Görselleştirme
                axes[i, 0].imshow(np.rot90(data[mx, :, :]), cmap='gray')
                axes[i, 1].imshow(np.rot90(data[:, my, :]), cmap='gray')
                axes[i, 2].imshow(np.rot90(data[:, :, mz]), cmap='gray')
                
                # Hasta Bilgi Kartı 
                gap_turu = "Hızlanmış" if row['Tahmin'] > row['age'] else "Yavaş"
                info_text = (f"ID: {row['subject_id']}\nCinsiyet: {row['sex']}\n"
                             f"Gerçek: {row['age']:.1f}\nTahmin: {row['Tahmin']:.1f}\n"
                             f"GAP: {row['Tahmin']-row['age']:.2f}\n"
                             f"({gap_turu} Yaşlanma)\n\n"
                             f"SITE: {row['site']}\nSCANNER: {row['scanner']}")
                
                axes[i, 0].set_ylabel(info_text, fontsize=9, fontweight='bold', labelpad=15, rotation=0, ha='right')

            if i == 0:
                for j in range(3): axes[i, j].set_title(planes[j], fontsize=14, fontweight='bold')

    for ax in axes.flatten(): ax.set_xticks([]); ax.set_yticks([])
    
    plt.suptitle("Yapay Zekanın En Çok Yanıldığı 5 Vaka (Outlier Analizi)", fontsize=20, y=1.01)
    plt.savefig("yasli_vaka_analizi.pdf", format='pdf', bbox_inches='tight', dpi=300)
    print("Grafik 'yasli_vaka_analizi.pdf' adıyla kaydedildi.")
    plt.tight_layout()
    plt.show()

# Terminale özet tablo
print("\n" + "="*80)
print("RAPOR İÇİN 'EN YÜKSEK HATALI HASTALAR' ÖZET TABLOSU")
print("="*80)
cols_to_show = ['subject_id', 'age', 'Tahmin', 'Hata', 'sex', 'scanner', 'site']
print(worst_cases[cols_to_show].to_string(index=False))
print("="*80)