import pandas as pd
import numpy as np
import zipfile
import nibabel as nib
import matplotlib.pyplot as plt
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. VERİ YÜKLEME VE ÖN HAZIRLIK
zip_path = r'C:\Users\pc\Desktop\Sınavlar\uyz\archive.zip'
print("1. Zip taranıyor ve 20-40 yaş grubu filtreleniyor...")

with zipfile.ZipFile(zip_path, 'r') as z:
    z_list = z.namelist()
    available_subjects = []
    for f in z_list:
        if f.endswith('.nii') and 'IXI' in f:
            clean_id = f.split('/')[-1].split('_')[0].replace('sub-', '')
            available_subjects.append({'id': clean_id, 'full_path': f})
    
    df_available = pd.DataFrame(available_subjects).drop_duplicates('id')
    
    thick_f = [f for f in z_list if 'thickness' in f and f.endswith('.csv')][0]
    subj_f = [f for f in z_list if 'subjects' in f and f.endswith('.csv')][0]
    with z.open(thick_f) as f: df_thick = pd.read_csv(f)
    with z.open(subj_f) as f: df_subj = pd.read_csv(f)

# 2. VERİ BİRLEŞTİRME VE MODEL EĞİTİMİ
df_thick['hemi_region'] = df_thick['hemi'] + '_' + df_thick['region']
df_piv = pd.pivot_table(df_thick, index='subject_id', columns='hemi_region', values='mean_thickness_weighted', aggfunc='mean').reset_index()

df_final = pd.merge(df_piv, df_available, left_on='subject_id', right_on='id')
df_final = pd.merge(df_final, df_subj, on='subject_id')
df_final = df_final[df_final['age'] > 0].fillna(df_final.mean(numeric_only=True))

X = df_final[df_piv.columns].drop(columns=['subject_id'])
y = df_final['age']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_train, y_train)
preds = model.predict(X_test)

# 3. 20-40 YAŞ ARASI EN YÜKSEK HATALARI BULMA
res = df_final.loc[X_test.index].copy()
res['Tahmin'] = preds
res['Hata'] = abs(res['Tahmin'] - res['age'])

# Sadece 20-40 yaş aralığını filtreleme
genc_yas_hatalilar = res[(res['age'] >= 20) & (res['age'] <= 40)]
worst_5_genc = genc_yas_hatalilar.sort_values('Hata', ascending=False).head(5)

# 4. GÖRSELLEŞTİRME VE PDF KAYDETME
if worst_5_genc.empty:
    print("HATA: Test setinde bu yaş aralığında hasta bulunamadı!")
else:
    print(f"2. 20-40 yaş arası en hatalı {len(worst_5_genc)} vaka çiziliyor...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        fig, axes = plt.subplots(5, 3, figsize=(16, 25))
        planes = ["Sagittal (X)", "Coronal (Y)", "Axial (Z)"]

        for i, (idx, row) in enumerate(worst_5_genc.iterrows()):
            with z.open(row['full_path']) as f:
                content = io.BytesIO(f.read())
                img = nib.Nifti1Image.from_file_map({'header': nib.FileHolder(fileobj=content), 
                                                      'image': nib.FileHolder(fileobj=content)})
                data = img.get_fdata()
                mx, my, mz = data.shape[0]//2, data.shape[1]//2, data.shape[2]//2
                
                # 3 Düzlem Çizimi
                axes[i, 0].imshow(np.rot90(data[mx, :, :]), cmap='gray')
                axes[i, 1].imshow(np.rot90(data[:, my, :]), cmap='gray')
                axes[i, 2].imshow(np.rot90(data[:, :, mz]), cmap='gray')
                
                # Bilgi Kartı
                gap_durumu = "HIZLANMIŞ" if row['Tahmin'] > row['age'] else "YAVAŞ"
                info = (f"ID: {row['subject_id']}\nCinsiyet: {row['sex']}\n"
                        f"Gerçek: {row['age']:.1f}\nTahmin: {row['Tahmin']:.1f}\n"
                        f"Hata: {row['Hata']:.2f}\n({gap_durumu})\n"
                        f"Site: {row['site']}\nScanner: {row['scanner']}")
                axes[i, 0].set_ylabel(info, fontsize=10, fontweight='bold', rotation=0, ha='right', labelpad=25)

        for j in range(3): axes[0, j].set_title(planes[j], fontsize=14, fontweight='bold')
        for ax in axes.flatten(): ax.set_xticks([]); ax.set_yticks([])
        
        plt.suptitle("20-40 Yaş Arası En Büyük Tahmin Hataları Analizi", fontsize=18, y=1.01)
        plt.tight_layout()
        plt.savefig("genc_yas_vaka_analizi.pdf", format='pdf', bbox_inches='tight', dpi=300)
        print("Grafik 'genc_yas_vaka_analizi.pdf' adıyla kaydedildi.")
        
        plt.show()

    print("\n--- 20-40 YAŞ GRUBU EN HATALI 5 HASTA ---")
    print(worst_5_genc[['subject_id', 'age', 'Tahmin', 'Hata', 'sex', 'scanner', 'site']].to_string(index=False))