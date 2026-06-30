import pandas as pd
import zipfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. VERİLERİ ZİP'TEN YÜKLEME
zip_path = r'C:\Users\pc\Desktop\Sınavlar\uyz\archive.zip'
print("1. Veriler okunuyor ve makine öğrenmesi için düzenleniyor...")

with zipfile.ZipFile(zip_path, 'r') as z:
    dosyalar = z.namelist()
    thick_file = [f for f in dosyalar if 'thickness' in f and f.endswith('.csv')][0]
    subj_file = [f for f in dosyalar if 'subjects' in f and f.endswith('.csv')][0]
    
    with z.open(thick_file) as f:
        df_thickness = pd.read_csv(f)
    with z.open(subj_file) as f:
        df_subjects = pd.read_csv(f)

# 2. VERİYİ DÜZENLEME VE PIVOT
df_thickness['hemi_region'] = df_thickness['hemi'] + '_' + df_thickness['region']
df_pivot = pd.pivot_table(
    df_thickness, index='subject_id', columns='hemi_region', 
    values='mean_thickness_weighted', aggfunc='mean'
).reset_index()

# 3. YAŞ VERİSİ VE EKSİK VERİ TAMAMLAMA
df_merged = pd.merge(df_pivot, df_subjects[['subject_id', 'age']], on='subject_id')
sayisal_sutunlar = df_merged.select_dtypes(include=['float64', 'int64']).columns
df_clean = df_merged.copy()
df_clean[sayisal_sutunlar] = df_clean[sayisal_sutunlar].fillna(df_clean[sayisal_sutunlar].mean())

# 4. X VE y DEĞERLERİNİ BELİRLEME
y = df_clean['age'] 
X = df_clean.drop(columns=['subject_id', 'age']) 

# 5. %80 EĞİTİM, %20 TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# İŞTE YENİ EKLENEN KISIM: HYPERPARAMETER TUNING (İNCE AYAR)
# ---------------------------------------------------------
print("\n2. YZ'nin ayar düğmeleri optimize ediliyor (Bu işlem bilgisayarını biraz yorabilir, 1-2 dakika sürebilir!)...")

# Denenecek ayar kombinasyonları
param_grid = {
    'n_estimators': [100, 200, 300],        # Ağaç sayısı
    'max_depth': [None, 10, 20],            # Ağaç derinliği
    'min_samples_split': [2, 5, 10]         # Bölünme hassasiyeti
}

# Modeli ve deneyiciyi (GridSearch) kuruyoruz
rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=rf_model, 
    param_grid=param_grid, 
    cv=3, # Veriyi 3'e katlayıp çapraz doğrulama yapar
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, # İşlemcinin (CPU) tüm çekirdeklerini kullanır (Hızlı bitmesi için)
    verbose=1
)

# En iyi ayarları bulması için eğitime başlıyoruz
grid_search.fit(X_train, y_train)

# En zeki modeli (best_estimator) seçiyoruz
en_iyi_model = grid_search.best_estimator_

print(f"\n-> Bulunan En İyi Ayarlar: {grid_search.best_params_}")

# 6. YENİ MODELLE TAHMİN YAPMA
beyin_yaslari = en_iyi_model.predict(X_test)
yeni_mae = mean_absolute_error(y_test, beyin_yaslari)

print(f"\n=====================================")
print(f"ESKİ HATA PAYI: 12.16 yıl")
print(f"YENİ (OPTİMİZE EDİLMİŞ) HATA PAYI: {yeni_mae:.2f} yıl")
print(f"=====================================")

# 7. GÖRSELLEŞTİRME
plt.figure(figsize=(9, 6))
plt.scatter(y_test, beyin_yaslari, alpha=0.7, color='indigo', edgecolors='white', s=60)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Kusursuz Tahmin')

plt.title('Chronological Age vs. Predicted Brain Age (Optimized Model)', fontsize=14)
plt.xlabel('Gerçek Kimlik Yaşı', fontsize=12)
plt.ylabel('YZ Tahmini (Brain Age)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()