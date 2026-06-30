import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# 1. VERİ YÜKLEME
zip_path = r'C:\Users\pc\Desktop\Sınavlar\uyz\archive.zip'
print("1. Veriler titizlikle işleniyor...")

with zipfile.ZipFile(zip_path, 'r') as z:
    z_list = z.namelist()
    thick_f = [f for f in z_list if 'thickness' in f and f.endswith('.csv')][0]
    subj_f = [f for f in z_list if 'subjects' in f and f.endswith('.csv')][0]
    with z.open(thick_f) as f: df_thick = pd.read_csv(f)
    with z.open(subj_f) as f: df_subj = pd.read_csv(f)

# Ön işleme
df_thick['hemi_region'] = df_thick['hemi'] + '_' + df_thick['region']
df_piv = pd.pivot_table(df_thick, index='subject_id', columns='hemi_region', values='mean_thickness_weighted', aggfunc='mean').reset_index()
df_m = pd.merge(df_piv, df_subj[['subject_id', 'age']], on='subject_id')
df_m = df_m[df_m['age'] > 0].fillna(df_m.mean(numeric_only=True))

X = df_m[df_piv.columns].drop(columns=['subject_id'])
y = df_m['age']

# Ölçeklendirme (SVR ve MLP için şart)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. MODELLERİN TANIMLANMASI
models = {
    "Lineer (Ridge)": Ridge(),
    "SVR (Vektör Destek)": SVR(kernel='rbf'),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
    "Yapay Sinir Ağları (MLP)": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

# 3. ANALİZ DÖNGÜSÜ
print("2. 5 Model için detaylı metrikler hesaplanıyor...")
res_list = []
age_perf = []

for name, model in models.items():
    model.fit(X_train, y_train)
    p_train = model.predict(X_train)
    p_test = model.predict(X_test)
    
    # Metrikler
    m_train = mean_absolute_error(y_train, p_train)
    m_test = mean_absolute_error(y_test, p_test)
    
    # Overfitting Skoru (Fark ne kadar çoksa o kadar kötü)
    of_score = m_test - m_train
    
    # Yaş Gruplarına Göre Performans (Point 3)
    temp_df = pd.DataFrame({'age': y_test, 'pred': p_test})
    temp_df['Grup'] = pd.cut(temp_df['age'], bins=[20, 40, 60, 90], labels=['Genç', 'Orta', 'İleri'])
    group_mae = temp_df.groupby('Grup', observed=False).apply(lambda x: mean_absolute_error(x['age'], x['pred']))
    
    # Açıklanabilirlik Skoru (Manuel Atama - Point 2)
    # 5: Cam Kutu, 1: Kara Kutu
    exp_score = 5 if "Lineer" in name else (4 if "Random" in name else (2 if "SVR" in name or "Gradient" in name else 1))
    
    res_list.append({
        "Model": name, "Train_MAE": m_train, "Test_MAE": m_test, 
        "Overfit": of_score, "Açıklanabilirlik": exp_score
    })
    
    for g, val in group_mae.items():
        age_perf.append({"Model": name, "Grup": g, "MAE": val})

df_res = pd.DataFrame(res_list)
df_age = pd.DataFrame(age_perf)

# 4. GÖRSEL ŞOV (SURPRISE DASHBOARD)
fig = plt.figure(figsize=(20, 15))
plt.suptitle("MODEL STRATEJİ ANALİZ PANELİ: NEDEN RANDOM FOREST?", fontsize=24, fontweight='bold', y=0.98)

# --- GRAFİK 1: AŞIRI ÖĞRENME (Overfitting) TUZAĞI ---
ax1 = plt.subplot(2, 2, 1)
df_melt = df_res.melt(id_vars='Model', value_vars=['Train_MAE', 'Test_MAE'], var_name='Set', value_name='MAE')
sns.barplot(data=df_melt, x='MAE', y='Model', hue='Set', palette=['#2ecc71', '#e74c3c'], ax=ax1)
ax1.set_title("1. Overfitting Analizi (Eğitim vs Test Hatası)", fontsize=15, fontweight='bold')
ax1.set_xlabel("Hata (Düşük İyidir)")

# --- GRAFİK 2: CAM KUTU VS KARA KUTU ---
ax2 = plt.subplot(2, 2, 2)
sns.scatterplot(data=df_res, x='Açıklanabilirlik', y='Test_MAE', hue='Model', s=400, style='Model', ax=ax2)
for i in range(df_res.shape[0]):
    ax2.text(df_res.Açıklanabilirlik[i]+0.1, df_res.Test_MAE[i], df_res.Model[i], fontweight='bold')
ax2.set_title("2. Açıklanabilirlik vs. Tahmin Başarısı", fontsize=15, fontweight='bold')
ax2.set_xlabel("Açıklanabilirlik (1: Kara Kutu, 5: Cam Kutu)")
ax2.set_ylabel("Test Hatası (MAE)")
ax2.set_xlim(0, 6)

# --- GRAFİK 3: DOĞRUSAL OLMAYAN ESNEKLİK (Yaş Grupları) ---
ax3 = plt.subplot(2, 2, 3)
sns.lineplot(data=df_age, x='Grup', y='MAE', hue='Model', marker='o', linewidth=3, ax=ax3)
ax3.set_title("3. Yaş Gruplarına Göre Kararlılık (Stabilite)", fontsize=15, fontweight='bold')
ax3.set_ylabel("Hata (Yıl)")
ax3.grid(True, linestyle='--', alpha=0.6)

# --- GRAFİK 4: KARAR MATRİSİ (Özet Isı Haritası) ---
ax4 = plt.subplot(2, 2, 4)
df_norm = df_res.set_index('Model')[['Test_MAE', 'Overfit', 'Açıklanabilirlik']]
# Normalize ederek (0-1 arası) kıyaslayalım
df_norm = (df_norm - df_norm.min()) / (df_norm.max() - df_norm.min())
sns.heatmap(df_norm, annot=True, cmap="YlGnBu", ax=ax4)
ax4.set_title("4. Karar Matrisi (Normalize Edilmiş Skorlar)", fontsize=15, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("5_Model_Stratejik_Karsilastirma.pdf", bbox_inches='tight')
plt.savefig("5_Model_Stratejik_Karsilastirma.png", dpi=300)
plt.show()

print("\n--- MODEL STRATEJİ TABLOSU ---")
print(df_res.sort_values("Test_MAE").to_string(index=False))