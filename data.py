import pandas as pd
import zipfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. VERİLERİ ZİP'TEN YÜKLEME
zip_path = r'C:\Users\pc\Desktop\Sınavlar\uyz\archive.zip'
print("1. Veriler zip dosyasının içinden okunuyor...")

try:
    with zipfile.ZipFile(zip_path, 'r') as z:
        dosyalar = z.namelist()
        thick_file = [f for f in dosyalar if 'thickness' in f and f.endswith('.csv')][0]
        subj_file = [f for f in dosyalar if 'subjects' in f and f.endswith('.csv')][0]
        
        with z.open(thick_file) as f:
            df_thickness = pd.read_csv(f)
            
        with z.open(subj_file) as f:
            df_subjects = pd.read_csv(f)
    print("-> Veriler başarıyla çekildi!")
except Exception as e:
    print(f"HATA OLUŞTU: {e}")
    exit()

# 2. VERİYİ DÜZENLEME
print("2. Veriler makine öğrenmesi için düzenleniyor...")
df_thickness['hemi_region'] = df_thickness['hemi'] + '_' + df_thickness['region']

df_pivot = pd.pivot_table(
    df_thickness,
    index='subject_id', 
    columns='hemi_region', 
    values='mean_thickness_weighted',
    aggfunc='mean'
).reset_index()

# 3. YAŞ VERİSİ İLE BİRLEŞTİRME VE EKSİK VERİLERİ ONARMA 
print("3. Yaş verileri eşleştiriliyor ve eksik bölgeler onarılıyor...")
df_merged = pd.merge(df_pivot, df_subjects[['subject_id', 'age']], on='subject_id')
print(f"-> Eşleşen ham hasta sayısı: {len(df_merged)}")

# Eksikleri silmek (dropna) yerine ortalama ile dolduruyoruz (Data Imputation)
sayisal_sutunlar = df_merged.select_dtypes(include=['float64', 'int64']).columns
df_clean = df_merged.copy()
df_clean[sayisal_sutunlar] = df_clean[sayisal_sutunlar].fillna(df_clean[sayisal_sutunlar].mean())

print(f"-> Eğitime hazır toplam hasta sayısı: {len(df_clean)}")

# 4. X VE Y DEĞERLERİNİ BELİRLEME
y = df_clean['age'] 
X = df_clean.drop(columns=['subject_id', 'age']) 

# 5. %80 EĞİTİM, %20 TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"4. Yapay Zeka {len(X_train)} hasta ile eğitilecek, {len(X_test)} hasta ile test edilecek.")

# 6. MODELİ KURMA VE EĞİTME
print("\n5. Yapay Zeka yaşlanma kalıplarını öğreniyor... (Bu işlem birkaç saniye sürebilir)")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train) 

# 7. TAHMİN YAPMA VE BAŞARIYI ÖLÇME
beyin_yaslari = model.predict(X_test)
mae = mean_absolute_error(y_test, beyin_yaslari)

print(f"\n=====================================")
print(f"SONUÇ: YZ'nin Ortalama Hata Payı (MAE) = {mae:.2f} yıl")
print(f"=====================================")

# 8. GÖRSELLEŞTİRME
plt.figure(figsize=(9, 6))
plt.scatter(y_test, beyin_yaslari, alpha=0.7, color='teal', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Kusursuz Tahmin')

plt.title('Chronological Age vs. Predicted Brain Age', fontsize=14)
plt.xlabel('Gerçek Kimlik Yaşı (Chronological Age)', fontsize=12)
plt.ylabel('Yapay Zekanın Tahmini (Brain Age)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()