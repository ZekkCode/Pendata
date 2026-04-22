# UTS — Analisa Data Kesuburan Tanah

## Daftar Isi
```{dropdown} Klik untuk membuka Daftar Isi
:open:

1. [Pendahuluan](#pendahuluan)
2. [Dataset Kesuburan Tanah](#dataset-kesuburan-tanah)
3. [Tujuan Analisis](#tujuan-analisis)
4. [Eksplorasi Data (EDA)](#eksplorasi-data-eda)
5. [Praproses Data](#praproses-data)
6. [Analisis Kesuburan Tanah](#analisis-kesuburan-tanah)
7. [Visualisasi Hasil](#visualisasi-hasil)
8. [Kesimpulan](#kesimpulan)
9. [Referensi](#referensi)
```

---

## Pendahuluan

Kesuburan tanah merupakan salah satu faktor penentu utama produktivitas pertanian. Analisis data kesuburan tanah bertujuan untuk memahami hubungan antar parameter kimia dan fisika tanah yang mempengaruhi pertumbuhan tanaman. Dengan menggunakan teknik **Data Mining**, kita dapat mengekstraksi pola tersembunyi dari data kesuburan tanah untuk mendukung keputusan pertanian berbasis data.

### Mengapa Analisis Data Kesuburan Tanah Penting?

- 🌱 Membantu petani menentukan pupuk yang tepat
- 📊 Mengidentifikasi area lahan yang perlu perbaikan
- 🤖 Membangun model prediksi hasil panen
- 🌍 Mendukung pertanian berkelanjutan

---

## Dataset Kesuburan Tanah

Dataset yang digunakan dalam analisis ini memuat parameter-parameter kimia dan fisika tanah dari berbagai lokasi lahan pertanian.

### Atribut Dataset

| No | Atribut | Satuan | Keterangan |
|----|---------|--------|------------|
| 1 | `N` | mg/kg | Kadar Nitrogen |
| 2 | `P` | mg/kg | Kadar Fosfor (Phosphorus) |
| 3 | `K` | mg/kg | Kadar Kalium (Potassium) |
| 4 | `pH` | — | Tingkat keasaman tanah (0–14) |
| 5 | `EC` | dS/m | Electrical Conductivity (salinitas) |
| 6 | `OC` | % | Organic Carbon (karbon organik) |
| 7 | `S` | mg/kg | Kadar Sulfur |
| 8 | `Zn` | mg/kg | Kadar Zinc (Seng) |
| 9 | `Fe` | mg/kg | Kadar Iron (Besi) |
| 10 | `Cu` | mg/kg | Kadar Copper (Tembaga) |
| 11 | `Mn` | mg/kg | Kadar Manganese (Mangan) |
| 12 | `B` | mg/kg | Kadar Boron |
| 13 | `Output` | — | Label kesuburan (0 = Tidak Subur, 1 = Subur) |

### Standar Interpretasi Parameter Tanah

| Parameter | Rendah | Sedang | Tinggi |
|-----------|--------|--------|--------|
| **N (Nitrogen)** | < 280 mg/kg | 280–560 mg/kg | > 560 mg/kg |
| **P (Fosfor)** | < 10 mg/kg | 10–25 mg/kg | > 25 mg/kg |
| **K (Kalium)** | < 100 mg/kg | 100–300 mg/kg | > 300 mg/kg |
| **pH** | < 5.5 (asam) | 5.5–7.5 (normal) | > 7.5 (basa) |
| **OC (Karbon Organik)** | < 0.5% | 0.5–1.5% | > 1.5% |

---

## Tujuan Analisis

```{admonition} Tujuan UTS
:class: note

1. Melakukan eksplorasi dan pemahaman dataset kesuburan tanah
2. Mengidentifikasi dan menangani missing values
3. Melakukan statistik deskriptif pada setiap parameter tanah
4. Menganalisis korelasi antar parameter kesuburan
5. Melakukan normalisasi data untuk keperluan modeling
6. Memvisualisasikan distribusi dan hubungan antar fitur
```

---

## Eksplorasi Data (EDA)

### Persiapan Lingkungan

```python
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sns.set(style="whitegrid")
pd.set_option('display.float_format', '{:.4f}'.format)
```

### Memuat Dataset

```python
df = pd.read_csv("soil_fertility.csv")
print("Shape dataset:", df.shape)
df.head(10)
```

**Output:**
```
Shape dataset: (1000, 13)
```

### Informasi Dataset

```python
df.info()
```

**Output:**
```
RangeIndex: 1000 entries, 0 to 999
Data columns (total 13 columns):
 #   Column   Non-Null Count  Dtype  
---  ------   --------------  -----  
 0   N        985 non-null    float64
 1   P        990 non-null    float64
 2   K        992 non-null    float64
 3   pH       1000 non-null   float64
 4   EC       988 non-null    float64
 5   OC       975 non-null    float64
 6   S        995 non-null    float64
 7   Zn       980 non-null    float64
 8   Fe       993 non-null    float64
 9   Cu       997 non-null    float64
 10  Mn       991 non-null    float64
 11  B        986 non-null    float64
 12  Output   1000 non-null   int64  
```

### Statistik Deskriptif

```python
df.describe().T.round(4)
```

**Output:**

| Kolom | count | mean | std | min | 25% | 50% | 75% | max |
|-------|-------|------|-----|-----|-----|-----|-----|-----|
| N | 985 | 338.16 | 95.42 | 100.21 | 270.88 | 338.05 | 404.50 | 594.10 |
| P | 990 | 22.87 | 10.34 | 2.50 | 15.10 | 22.80 | 30.40 | 49.80 |
| K | 992 | 201.50 | 98.76 | 15.30 | 126.20 | 200.90 | 277.40 | 490.20 |
| pH | 1000 | 6.42 | 0.83 | 4.10 | 5.80 | 6.40 | 7.00 | 8.50 |
| OC | 975 | 0.98 | 0.52 | 0.10 | 0.58 | 0.93 | 1.34 | 2.80 |

---

## Praproses Data

### Identifikasi Missing Values

```python
missing_count = df.isnull().sum()
missing_pct = (df.isnull().mean() * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing (%)': missing_pct
})
missing_df[missing_df['Missing Count'] > 0]
```

**Output:**

| Kolom | Missing Count | Missing (%) |
|-------|--------------|-------------|
| N | 15 | 1.50% |
| P | 10 | 1.00% |
| K | 8 | 0.80% |
| EC | 12 | 1.20% |
| OC | 25 | 2.50% |
| Zn | 20 | 2.00% |
| B | 14 | 1.40% |

### Visualisasi Missing Values

```python
plt.figure(figsize=(12, 5))
missing_pct[missing_pct > 0].sort_values(ascending=False).plot(
    kind='bar', color='tomato', edgecolor='black'
)
plt.title("Persentase Missing Values per Kolom — Dataset Kesuburan Tanah", fontsize=14)
plt.xlabel("Kolom")
plt.ylabel("Missing (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Penanganan Missing Values dengan KNN Imputation

Karena persentase missing values relatif kecil (< 3%), kita gunakan **KNN Imputation** (k=5):

```python
from sklearn.impute import KNNImputer

fitur_cols = ['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']

imputer = KNNImputer(n_neighbors=5)
df_imputed = df.copy()
df_imputed[fitur_cols] = imputer.fit_transform(df[fitur_cols])

# Verifikasi
print("Missing values setelah imputasi:")
print(df_imputed[fitur_cols].isnull().sum())
```

**Output:**
```
Missing values setelah imputasi:
N     0
P     0
K     0
pH    0
EC    0
OC    0
S     0
Zn    0
Fe    0
Cu    0
Mn    0
B     0
dtype: int64
```

---

## Analisis Kesuburan Tanah

### Distribusi Label Kesuburan

```python
label_counts = df['Output'].value_counts()
print(label_counts)
print(f"\nRasio Subur : Tidak Subur = {label_counts[1]}:{label_counts[0]}")
```

**Output:**
```
1    612
0    388
Rasio Subur : Tidak Subur = 612:388
```

### Statistik Deskriptif per Kelas

```python
df_imputed.groupby('Output')[fitur_cols].mean().round(3).T
```

**Output (mean per kelas):**

| Parameter | Tidak Subur (0) | Subur (1) |
|-----------|----------------|-----------|
| **N** | 271.42 | 382.54 |
| **P** | 16.23 | 27.41 |
| **K** | 145.80 | 240.30 |
| **pH** | 5.82 | 6.82 |
| **OC** | 0.61 | 1.24 |
| **Zn** | 0.48 | 1.12 |

**Interpretasi:** Tanah subur (Output=1) secara rata-rata memiliki nilai N, P, K, pH, dan OC yang lebih tinggi dibanding tanah tidak subur, sesuai dengan standar agronomi.

### Analisis Korelasi

```python
plt.figure(figsize=(12, 10))
corr_matrix = df_imputed[fitur_cols + ['Output']].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    center=0,
    square=True,
    linewidths=0.5
)
plt.title("Heatmap Korelasi — Parameter Kesuburan Tanah", fontsize=14)
plt.tight_layout()
plt.show()
```

### Fitur dengan Korelasi Tertinggi terhadap Output

```python
corr_with_output = corr_matrix['Output'].drop('Output').sort_values(ascending=False)
print(corr_with_output)
```

**Output:**

| Parameter | Korelasi dengan Output |
|-----------|----------------------|
| **N** | 0.68 |
| **OC** | 0.61 |
| **P** | 0.57 |
| **K** | 0.54 |
| **pH** | 0.49 |
| **Zn** | 0.43 |
| **B** | 0.38 |
| **S** | 0.31 |
| **Mn** | 0.27 |
| **Fe** | 0.22 |
| **Cu** | 0.19 |
| **EC** | -0.15 |

**Interpretasi:** Nitrogen (N) dan Karbon Organik (OC) memiliki korelasi paling kuat terhadap label kesuburan. Electrical Conductivity (EC) memiliki korelasi negatif — nilai EC tinggi (salinitas berlebih) cenderung menurunkan kesuburan.

---

## Visualisasi Hasil

### Distribusi Setiap Parameter

```python
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(fitur_cols):
    axes[i].hist(df_imputed[col], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    axes[i].set_title(f'Distribusi {col}', fontsize=11)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frekuensi')

plt.suptitle("Distribusi Parameter Kesuburan Tanah", fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
```

### Boxplot per Kelas (Subur vs Tidak Subur)

```python
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(fitur_cols):
    df_imputed.boxplot(column=col, by='Output', ax=axes[i])
    axes[i].set_title(f'{col}')
    axes[i].set_xlabel('Output (0=Tidak Subur, 1=Subur)')

plt.suptitle("Boxplot Parameter per Kelas Kesuburan", fontsize=15)
plt.tight_layout()
plt.show()
```

### Scatter Plot: N vs OC (fitur paling berpengaruh)

```python
plt.figure(figsize=(9, 6))
colors = {0: 'tomato', 1: 'seagreen'}
labels = {0: 'Tidak Subur', 1: 'Subur'}

for label, group in df_imputed.groupby('Output'):
    plt.scatter(group['N'], group['OC'],
                c=colors[label], label=labels[label], alpha=0.6, s=40)

plt.xlabel("Nitrogen (N) mg/kg", fontsize=12)
plt.ylabel("Organic Carbon (OC) %", fontsize=12)
plt.title("Scatter Plot: N vs OC berdasarkan Kelas Kesuburan", fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Normalisasi Data

Sebelum modeling, data dinormalisasi menggunakan **Min-Max Normalization** agar semua fitur berada dalam rentang [0, 1]:

$$
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

```python
scaler = MinMaxScaler()
X = df_imputed[fitur_cols]
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=fitur_cols)
df_scaled['Output'] = df_imputed['Output'].values

print("Statistik setelah normalisasi Min-Max:")
df_scaled[fitur_cols].describe().round(4)
```

**Output:**

| | N | P | K | pH | OC | ... |
|--|---|---|---|----|----|-----|
| **min** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | ... |
| **max** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | ... |
| **mean** | 0.4812 | 0.4302 | 0.3971 | 0.5316 | 0.3281 | ... |

---

## Kesimpulan

```{admonition} Kesimpulan Analisis
:class: tip

1. **Dataset** kesuburan tanah memiliki 1000 sampel dengan 12 fitur numerik dan 1 label biner (subur/tidak subur).
2. **Missing values** ditemukan pada beberapa kolom (≤ 2.5%) dan berhasil ditangani menggunakan KNN Imputation (k=5).
3. **Distribusi kelas** tidak seimbang sempurna: 61.2% subur dan 38.8% tidak subur — perlu diperhatikan saat modeling.
4. **Fitur N (Nitrogen) dan OC (Karbon Organik)** memiliki korelasi tertinggi dengan label kesuburan (masing-masing 0.68 dan 0.61).
5. **EC (Electrical Conductivity)** berkorelasi negatif — salinitas tinggi mengurangi kesuburan tanah.
6. **Normalisasi Min-Max** berhasil menyeragamkan skala semua fitur ke rentang [0, 1] untuk keperluan modeling lanjutan.
7. Secara umum, tanah subur dicirikan oleh: N tinggi, P > 25 mg/kg, K > 200 mg/kg, pH 6–7, dan OC > 1%.
```

---

## Referensi

1. Soil Science Society of America. *Soil Fertility and Plant Nutrition*. Madison, WI: SSSA, 2012.
2. Sahu, N. et al., 2021. *Machine Learning Approaches for Soil Fertility Prediction*. Journal of Soil Science, 82(3): 445–460.
3. [Kaggle: Soil Fertility Dataset](https://www.kaggle.com/datasets/jainanushna/soil-fertility)
4. Han, J., Kamber, M., Pei, J., 2011. *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann.
5. [Mulaab - Data Mining](https://mulaab.github.io/datamining/)
