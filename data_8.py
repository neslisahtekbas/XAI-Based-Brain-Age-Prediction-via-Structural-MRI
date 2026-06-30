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
print("1. Veriler ve hasta bilgileri harmanlanıyor...")

with zipfile.ZipFile(zip_path, 'r') as z:
    thick_f = [f for f in z.namelist() if 'thickness' in f and f.endswith('.csv')][0]
    subj_f = [f for f in z.namelist() if 'subjects' in f and f.endswith('.csv')][0]
    with z.open(thick_f) as f: df_thick = pd.read_csv(f)
    with z.open(subj_f) as f: df_subj = pd.read_csv(f)

# Ön işleme ve Birleştirme (Tüm demografik verileri koruyoruz)
df_thick['hemi_region'] = df_thick['hemi'] + '_' + df_thick['region']
df_piv = pd.pivot_table(df_thick, index='subject_id', columns='hemi_region', values='mean_thickness_weighted', aggfunc='mean').reset_index()

# subjects dosyasındaki tüm bilgileri (cinsiyet, site, scanner vb.) ekliyoruz
df_m = pd.merge(df_piv, df_subj, on='subject_id')
df_m = df_m[df_m['age'] > 0]
df_m = df_m.fillna(df_m.mean(numeric_only=True))

# Model Hazırlığı (Sadece sayısal beyin verilerini X'e alıyoruz)
X = df_m[df_piv.columns].drop(columns=['subject_id'])
y = df_m['age']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("2. Yapay Zeka eğitiliyor...")
model = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_train, y_train)
preds = model.predict(X_test)

# 2. EN BAŞARILI HASTALARI VE TÜM BİLGİLERİNİ SEÇME
test_indices = X_test.index
res = df_m.loc[test_indices].copy()
res['Tahmin'] = preds
res['Hata'] = abs(res['Tahmin'] - res['age'])

# 3 Yaş grubundan en başarılı 1'er kişiyi çek
best_cases = pd.concat([
    res[res['age']<=40].sort_values('Hata').head(1), 
    res[(res['age']>40)&(res['age']<=60)].sort_values('Hata').head(1), 
    res[res['age']>60].sort_values('Hata').head(1)
])

# 3. 9'LU PANEL VE DETAYLI XYZ ANALİZİ
print("3. 9 Kesitli Görsel ve Detaylı Hasta Tablosu Oluşturuluyor...")
with zipfile.ZipFile(zip_path, 'r') as z:
    z_list = z.namelist()
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    planes = ["Sagittal (X)", "Coronal (Y)", "Axial (Z)"]
    groups = ["GENÇ", "ORTA YAŞ", "İLERİ YAŞ"]

    for i, (idx, row) in enumerate(best_cases.iterrows()):
        f_name = [f for f in z_list if row['subject_id'] in f and f.endswith('.nii')][0]
        with z.open(f_name) as f:
            content = io.BytesIO(f.read())
            img = nib.Nifti1Image.from_file_map({'header': nib.FileHolder(fileobj=content), 'image': nib.FileHolder(fileobj=content)})
            data = img.get_fdata()
            
            mx, my, mz = data.shape[0]//2, data.shape[1]//2, data.shape[2]//2
            
            # 3 Düzlem Görselleştirme
            axes[i, 0].imshow(np.rot90(data[mx, :, :]), cmap='gray') # Sagittal
            axes[i, 1].imshow(np.rot90(data[:, my, :]), cmap='gray') # Coronal
            axes[i, 2].imshow(np.rot90(data[:, :, mz]), cmap='gray') # Axial
            
            # Sol tarafa hasta kimlik bilgilerini yazdır
            info_text = (f"GRUP: {groups[i]}\nID: {row['subject_id']}\nCinsiyet: {row['sex']}\n"
                         f"Gerçek Yaş: {row['age']:.1f}\nTahmin: {row['Tahmin']:.1f}\n"
                         f"Hata: {row['Hata']:.2f} yıl\n\n"
                         f"SCANNER: {row['scanner']}\nSITE: {row['site']}\n"
                         f"KOORDİNATLAR:\nX:{mx} Y:{my} Z:{mz}")
            
            axes[i, 0].set_ylabel(info_text, fontsize=10, fontweight='bold', labelpad=20, rotation=0, ha='right')

            if i == 0:
                for j in range(3): axes[i, j].set_title(planes[j], fontsize=14, fontweight='bold')

    for ax in axes.flatten(): ax.set_xticks([]); ax.set_yticks([])
    
    plt.suptitle("Yapay Zeka Beyin Yaşı Analiz Raporu: Çok Boyutlu Vaka İncelemesi", fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()

# Terminale de temiz bir özet basalım
print("\n" + "="*60)
print("RAPOR İÇİN HASTA ÖZET TABLOSU")
print("="*60)
cols_to_show = ['subject_id', 'age', 'Tahmin', 'Hata', 'sex', 'scanner', 'site']
print(best_cases[cols_to_show].to_string(index=False))
print("="*60)