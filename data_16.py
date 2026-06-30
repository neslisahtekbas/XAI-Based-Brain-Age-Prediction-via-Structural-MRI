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
print("1. Veriler yükleniyor ve 16 deneklik panel hazırlanıyor...")

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

# Veri Birleştirme
df_thick['hemi_region'] = df_thick['hemi'] + '_' + df_thick['region']
df_piv = pd.pivot_table(df_thick, index='subject_id', columns='hemi_region', values='mean_thickness_weighted', aggfunc='mean').reset_index()
df_final = pd.merge(df_piv, df_available, left_on='subject_id', right_on='id')
df_final = pd.merge(df_final, df_subj, on='subject_id')
df_final = df_final[df_final['age'] > 0].fillna(df_final.mean(numeric_only=True))

# Eğitim
X = df_final[df_piv.columns].drop(columns=['subject_id'])
y = df_final['age']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_train, y_train)
preds = model.predict(X_test)

# 2. RASTGELE 16 DENEK SEÇİMİ
res = df_final.loc[X_test.index].copy()
res['Tahmin'] = preds
res['Hata'] = abs(res['Tahmin'] - res['age'])

# Test setinden 16 tane denek al (Eğer test seti 16'dan küçükse hepsini al)
sample_size = min(16, len(res))
sample_16 = res.sample(sample_size, random_state=42)

# 3. GÖRSELLEŞTİRME (4x4 Izgara)
print(f"2. {sample_size} denek için 4x4 panel oluşturuluyor...")
with zipfile.ZipFile(zip_path, 'r') as z:
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes_flat = axes.flatten()

    for i, (idx, row) in enumerate(sample_16.iterrows()):
        with z.open(row['full_path']) as f:
            content = io.BytesIO(f.read())
            img = nib.Nifti1Image.from_file_map({'header': nib.FileHolder(fileobj=content), 
                                                 'image': nib.FileHolder(fileobj=content)})
            data = img.get_fdata()
            mz = data.shape[2] // 2 # Z=91 katmanı
            
            axes_flat[i].imshow(np.rot90(data[:, :, mz]), cmap='gray')
            
            # Bilgi Metni
            info = (f"ID: {row['subject_id']}\n"
                    f"G: {row['age']:.1f} | T: {row['Tahmin']:.1f}\n"
                    f"Hata: {row['Hata']:.2f} Yıl")
            
            axes_flat[i].set_title(info, fontsize=10, fontweight='bold')
            axes_flat[i].axis('off')

    # Boş kalan kareleri temizle (Eğer 16'dan az denek varsa)
    for j in range(i + 1, 16):
        axes_flat[j].axis('off')

    plt.suptitle("Model Tahmin Başarısı: 16 Farklı Denek Üzerinde Kesit Analizi", fontsize=22, y=1.02)
    plt.tight_layout()
    
    # PDF KAYDETME
    plt.savefig("16_denek_analiz_paneli.pdf", format='pdf', bbox_inches='tight', dpi=300)
    print("Panel '16_denek_analiz_paneli.pdf' adıyla başarıyla kaydedildi.")
    
    plt.show()