import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. VERİLERİ ZİP'TEN YÜKLEME VE DÜZENLEME (Artık buraları biliyorsun)
zip_path = r'C:\Users\pc\Desktop\Sınavlar\uyz\archive.zip'
print("1. Veriler okunuyor ve hazırlanıyor...")

with zipfile.ZipFile(zip_path, 'r') as z:
    dosyalar = z.namelist()
    thick_file = [f for f in dosyalar if 'thickness' in f and f.endswith('.csv')][0]
    subj_file = [f for f in dosyalar if 'subjects' in f and f.endswith('.csv')][0]
    
    with z.open(thick_file) as f:
        df_thickness = pd.read_csv(f)
    with z.open(subj_file) as f:
        df_subjects = pd.read_csv(f)

df_thickness['hemi_region'] = df_thickness['hemi'] + '_' + df_thickness['region']
df_pivot = pd.pivot_table(
    df_thickness, index='subject_id', columns='hemi_region', 
    values='mean_thickness_weighted', aggfunc='mean'
).reset_index()

df_merged = pd.merge(df_pivot, df_subjects[['subject_id', 'age']], on='subject_id')
sayisal_sutunlar = df_merged.select_dtypes(include=['float64', 'int64']).columns
df_clean = df_merged.copy()
df_clean[sayisal_sutunlar] = df_clean[sayisal_sutunlar].fillna(df_clean[sayisal_sutunlar].mean())

y = df_clean['age'] 
X = df_clean.drop(columns=['subject_id', 'age']) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. MODELİ EĞİTME (Az önce bulduğumuz en iyi ayarları kullanıyoruz)
print("2. Yapay Zeka eğitiliyor...")
model = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=2, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 3. İŞTE BÜYÜ O BURADA: SHAP ANALİZİ
# ---------------------------------------------------------
print("3. SHAP Analizi başlatılıyor (Yapay zekanın beyni taranıyor!)...")

# SHAP Explainer (Açıklayıcı) objesini kuruyoruz
explainer = shap.TreeExplainer(model)

# Test verilerimizdeki hastalar için SHAP değerlerini hesaplıyoruz
shap_values = explainer.shap_values(X_test)

# 4. GÖRSELLEŞTİRME (Beeswarm Grafiği)
print("\n-> SHAP grafiği ekrana getiriliyor! Lütfen grafiği incele.")
plt.figure(figsize=(10, 8))
plt.title("Hangi Beyin Bölgeleri Yaşlanmada En Etkili? (SHAP Analizi)")

# Bu komut o meşhur renkli, noktalı makale grafiğini çizer
shap.summary_plot(shap_values, X_test, show=False)

# Grafiğin ekrandan taşmaması için düzenleme
plt.tight_layout()
plt.show()