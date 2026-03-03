# DATA PREPARATION — PERTEMUAN 3
## Studi Kasus: Iris + Data Campuran (Mixed-Type)

| | |
|---|---|
| **Nama** | Zakaria Mujur Prasetyo |
| **NIM** | 240411100144 |

Dokumen ini melanjutkan materi Data Preparation dalam kerangka **CRISP-DM** yang mencakup:
identifikasi missing value, statistik deskriptif, encoding, scaling, **pengukuran jarak**, dan penanganan **data campuran (mixed-type)**.

---

## ✅ Tugas Pertemuan 3

```{admonition} Tugas yang Harus Diselesaikan
:class: important

Berikut tiga tugas utama pada Pertemuan 3 beserta status penyelesaiannya:

| No | Tugas | Status | Keterangan |
|:--:|-------|:------:|------------|
| 1 | **Mengukur Jarak** — ditempatkan di bawah bagian *Data Understanding* | ✅ Selesai | Euclidean, Manhattan, Spearman, Hamming pada data Iris (CSV & SQL) — lihat **Section 3.13–3.14** |
| 2 | **Buat/Cari Data Campuran** — mengandung tipe ordinal, numerik, kategorikal, dan biner | ✅ Selesai | Dataset **Bank Marketing** (`bank.csv` + PostgreSQL `BankMarketing`) — lihat **Section 3.15** |
| 3 | **Lakukan Pengukuran Jarak pada Data Campuran** tersebut | ✅ Selesai | 4 metrik jarak diterapkan di Orange pada data Bank Marketing — lihat **Section 3.15.5** |
```

> **File Orange Workflow:** dapat diunduh di {download}`DataCampuranBank.ows <DataCampuranPertemuan3/Bank Marketing/DataCampuranBank.ows>`

---

## 3.1 Konsep CRISP-DM

**CRISP-DM** (Cross-Industry Standard Process for Data Mining) adalah metodologi standar dalam proyek data mining yang terdiri dari 6 fase berurutan:

| No | Fase | Keterangan |
|----|------|------------|
| 1 | Business Understanding | Memahami tujuan bisnis dan kebutuhan analisis |
| 2 | Data Understanding | Eksplorasi awal data, statistik deskriptif |
| 3 | **Data Preparation** | Pembersihan, transformasi, seleksi fitur |
| 4 | Modeling | Membangun model machine learning |
| 5 | Evaluation | Mengevaluasi performa model |
| 6 | Deployment | Implementasi model ke sistem nyata |

> Pertemuan ini berfokus pada fase **Data Preparation** — fase paling kritis yang memakan 60–70% waktu proyek data mining.

---

## 3.2 Persiapan Lingkungan

Sebelum memulai analisis, kita impor library yang dibutuhkan. Setiap library memiliki peran khusus dalam proses data preparation.

```python
%matplotlib inline
import pandas as pd          # manipulasi dan analisis data tabular
import numpy as np           # komputasi numerik dan array
import matplotlib.pyplot as plt  # visualisasi data

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import pairwise_distances
```

| Library | Fungsi Utama |
|---------|-------------|
| `pandas` | Load CSV, manipulasi DataFrame, groupby, describe |
| `numpy` | Operasi array, kalkulasi jarak manual |
| `matplotlib` | Plot histogram, visualisasi distribusi |
| `StandardScaler` | Normalisasi fitur (mean=0, std=1) sebelum hitung jarak |
| `LabelEncoder` | Konversi label kategorikal ke numerik |
| `pairwise_distances` | Hitung distance matrix antar semua pasang data |

---

## 3.3 Memuat Dataset Awal

Dataset dimuat kembali untuk memastikan seluruh proses preparation dilakukan pada data mentah yang konsisten.

```python
df = pd.read_csv("IRIS.csv")
df.head()
```

**Output `df.head()`** — 5 baris pertama dataset Iris:

| | sepal_length | sepal_width | petal_length | petal_width | species |
|--|---|---|---|---|---|
| **0** | 5.1 | 3.5 | 1.4 | 0.2 | Iris-setosa |
| **1** | 4.9 | 3.0 | 1.4 | 0.2 | Iris-setosa |
| **2** | 4.7 | 3.2 | 1.3 | 0.2 | Iris-setosa |
| **3** | 4.6 | 3.1 | 1.5 | 0.2 | Iris-setosa |
| **4** | 5.0 | 3.6 | 1.4 | 0.2 | Iris-setosa |

Dataset ini berisi **150 baris** dan **5 kolom**, terdiri dari 4 fitur numerik dan 1 kolom target kategorikal.

---

## 3.4 Penjelasan: Fitur vs Kelas (Target)

Memahami perbedaan **fitur** dan **kelas** adalah dasar sebelum melakukan pemodelan supervised learning.

- **Fitur (features / attributes)** = kolom input yang menjadi karakteristik bunga, digunakan sebagai variabel independen (X).
- **Kelas (class / label / target)** = kolom output yang ingin diprediksi, merupakan variabel dependen (y).

**Tabel Identifikasi Kolom Dataset Iris:**

| Kolom | Tipe Data | Peran | Keterangan |
|-------|-----------|-------|------------|
| `sepal_length` | Numerik (float) | **Fitur** | Panjang kelopak luar / sepal (cm) |
| `sepal_width` | Numerik (float) | **Fitur** | Lebar kelopak luar / sepal (cm) |
| `petal_length` | Numerik (float) | **Fitur** | Panjang mahkota bunga / petal (cm) |
| `petal_width` | Numerik (float) | **Fitur** | Lebar mahkota bunga / petal (cm) |
| `species` | Kategorikal (string) | **Kelas (Target)** | Jenis bunga: *setosa*, *versicolor*, *virginica* |

✅ **Kesimpulan:** `sepal_length`, `sepal_width`, `petal_length`, `petal_width` → **fitur**.
Sedangkan `Iris-setosa`, `Iris-versicolor`, `Iris-virginica` → **kelas/label**.

> Jika membuat kolom `species_encoded`, itu hanya versi **numerik** dari kelas — bukan fitur baru.

---

## Pembersihan Data

---

## 3.5 Identifikasi Missing Value

Identifikasi missing value adalah langkah **pertama dan wajib** dalam data preparation. Data yang memiliki nilai kosong dapat menyebabkan error pada algoritma atau hasil analisis yang bias.

### 3.5.1 Jumlah Missing per Kolom

```python
missing_count = df.isnull().sum()
missing_count
```

### 3.5.2 Persentase Missing per Kolom

```python
missing_percent = (df.isnull().mean() * 100).round(2)
pd.DataFrame({'missing_count': missing_count, 'missing_%': missing_percent})
```

**Hasil Pengecekan Missing Value Dataset Iris:**

| Kolom | Missing Count | Missing % | Status |
|-------|:---:|:---:|:---:|
| `sepal_length` | 0 | 0.00% | ✅ Lengkap |
| `sepal_width` | 0 | 0.00% | ✅ Lengkap |
| `petal_length` | 0 | 0.00% | ✅ Lengkap |
| `petal_width` | 0 | 0.00% | ✅ Lengkap |
| `species` | 0 | 0.00% | ✅ Lengkap |

> Dataset Iris **tidak memiliki missing value**, sehingga tidak diperlukan proses imputasi (pengisian nilai kosong).

### 3.5.3 Menampilkan Baris yang Memiliki Missing (jika ada)

```python
rows_with_missing = df[df.isnull().any(axis=1)]
rows_with_missing.head()
```

---

## Statistik Deskriptif

---

## 3.5.4 Statistik Deskriptif per Fitur (Overall)

Statistik deskriptif memberikan gambaran umum distribusi data setiap fitur — ukuran pusat (mean, median) dan ukuran sebaran (std, min, max).

```python
numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df[numeric_cols].describe().T
```

**Ringkasan Statistik Deskriptif (150 data):**

| Fitur | count | mean | std | min | 25% | 50% | 75% | max |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| sepal_length | 150 | 5.843 | 0.828 | 4.3 | 5.1 | 5.80 | 6.4 | 7.9 |
| sepal_width | 150 | 3.054 | 0.434 | 2.0 | 2.8 | 3.00 | 3.3 | 4.4 |
| petal_length | 150 | 3.759 | 1.765 | 1.0 | 1.6 | 4.35 | 5.1 | 6.9 |
| petal_width | 150 | 1.199 | 0.763 | 0.1 | 0.3 | 1.30 | 1.8 | 2.5 |

### 3.5.5 Frekuensi Tiap Kelas

```python
df['species'].value_counts()
```

**Distribusi Kelas (Species):**

| Kelas | Jumlah | Persentase |
|-------|:---:|:---:|
| Iris-setosa | 50 | 33.3% |
| Iris-versicolor | 50 | 33.3% |
| Iris-virginica | 50 | 33.3% |

> Dataset Iris **seimbang** (*balanced*) — setiap kelas memiliki jumlah data yang sama (50 sampel), sehingga tidak diperlukan teknik resampling.

### 3.5.6 Statistik Deskriptif per Kelas (Ringkas)

```python
df.groupby('species')[numeric_cols].agg(['mean','std','min','max']).round(3)
```

**Statistik Mean per Kelas:**

| Kelas | sepal_length | sepal_width | petal_length | petal_width |
|-------|:---:|:---:|:---:|:---:|
| Iris-setosa | 5.006 | 3.418 | 1.464 | 0.244 |
| Iris-versicolor | 5.936 | 2.770 | 4.260 | 1.326 |
| Iris-virginica | 6.588 | 2.974 | 5.552 | 2.026 |

Tampilkan pairplot untuk melihat distribusi fitur per kelas secara visual:

```python
import matplotlib.pyplot as plt
import pandas as pd
pd.plotting.scatter_matrix(df[numeric_cols], figsize=(10, 8), c=df['species'].astype('category').cat.codes)
plt.suptitle('Pairplot Fitur Iris per Kelas')
plt.tight_layout()
plt.show()
```

![Pairplot Iris Dataset](Assets/Pertemuan_2/Pairplot.png)

![Scatter Plot Petal](Assets/Pertemuan_2/ScatterPlotPetal.png)

---

## Data Collecting

---

> 💡 **Catatan:** Setelah memahami statistik data (*Data Understanding*), langkah berikutnya adalah **pengukuran jarak** antar sampel. Dalam urutan CRISP-DM, pengukuran jarak dilakukan tepat setelah eksplorasi data — lihat **Section 3.13** untuk detail metrik dan implementasi.

## 3.11 Cara Collecting Data

Data collecting adalah proses mengumpulkan data **sebelum** preparation dimulai. Kualitas data yang dikumpulkan sangat menentukan kualitas model yang dihasilkan — prinsip *"garbage in, garbage out"*.

**Sumber Data yang Umum Digunakan:**

| Sumber | Contoh Format | Keterangan |
|--------|--------------|------------|
| File lokal | CSV, Excel, JSON | Cara paling umum, mudah diimpor ke Python/Orange |
| Database | MySQL, PostgreSQL | Data terstruktur dari sistem informasi |
| API/Web | REST API, JSON response | Data real-time dari layanan online |
| Sensor/IoT | Time-series, stream | Data dari perangkat fisik |
| Web scraping | HTML → CSV | Pengambilan data web (jika diizinkan) |

**Tahapan Umum Collecting:**

1. Tentukan kebutuhan — fitur apa, kelas apa, berapa banyak data
2. Ambil data — download file / query DB / panggil API
3. Simpan versi **raw** (mentah) sebelum dimodifikasi apapun
4. Buat **data dictionary** — dokumentasi arti kolom, satuan, tipe data
5. Baru masuk ke fase **data preparation**

**Contoh Data Dictionary untuk Dataset Iris:**

| Kolom | Tipe | Satuan | Nilai Unik | Keterangan |
|-------|------|--------|:----------:|------------|
| `sepal_length` | float | cm | kontinu | Panjang sepal bunga |
| `sepal_width` | float | cm | kontinu | Lebar sepal bunga |
| `petal_length` | float | cm | kontinu | Panjang petal bunga |
| `petal_width` | float | cm | kontinu | Lebar petal bunga |
| `species` | string | — | 3 | Kelas/label jenis bunga Iris |

---

## Menarik Data dari Database

---

## 3.12 Cara Menarik Data dari MySQL/PostgreSQL ke Orange

Orange dapat mengambil data langsung dari database relasional melalui widget **SQL Table**. Ini berguna ketika data disimpan di server database dan tidak tersedia sebagai file CSV.

### 3.12.1 Langkah Umum (Workflow Orange)

1. Buka **Orange Data Mining**
2. Dari panel widget, tambahkan: **SQL Table**
3. Pilih tipe database: **MySQL** atau **PostgreSQL**
4. Isi parameter koneksi
5. Pilih tabel atau tulis query SQL kustom
6. Sambungkan output ke widget: **Data Table** → **Select Columns** → **Impute** → **Normalize**

### 3.12.2 Contoh Parameter Koneksi

| Parameter | MySQL | PostgreSQL |
|-----------|-------|-----------|
| **Host** | `localhost` | `localhost` |
| **Port** | `3306` | `5432` |
| **Database** | `nama_db` | `nama_db` |
| **User** | `root` | `postgres` |
| **Password** | `(password Anda)` | `(password Anda)` |

### 3.12.3 Contoh Query SQL

```sql
SELECT sepal_length, sepal_width, petal_length, petal_width, species
FROM iris
WHERE sepal_length IS NOT NULL;
```

> Kalau widget **SQL Table** belum tersedia: buka **Options → Add-ons**, cari dan install add-on **Orange-SQL** atau yang mendukung koneksi database.

---

## Transformasi Data

---

## 3.6 Encoding Label

Karena algoritma machine learning memerlukan data numerik, maka label `species` bertipe string perlu dikonversi menjadi bentuk numerik menggunakan `LabelEncoder`.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['species'])
df.head()
```

**Mapping Encoding:**

| Label Asli | Encoded | Keterangan |
|-----------|:-------:|------------|
| `Iris-setosa` | **0** | Kelas pertama secara alfabet |
| `Iris-versicolor` | **1** | Kelas kedua |
| `Iris-virginica` | **2** | Kelas ketiga |

**Output `df.head()` setelah Encoding:**

| | sepal_length | sepal_width | petal_length | petal_width | species | species_encoded |
|--|---|---|---|---|---|:---:|
| **0** | 5.1 | 3.5 | 1.4 | 0.2 | Iris-setosa | 0 |
| **1** | 4.9 | 3.0 | 1.4 | 0.2 | Iris-setosa | 0 |
| **2** | 4.7 | 3.2 | 1.3 | 0.2 | Iris-setosa | 0 |
| **3** | 4.6 | 3.1 | 1.5 | 0.2 | Iris-setosa | 0 |
| **4** | 5.0 | 3.6 | 1.4 | 0.2 | Iris-setosa | 0 |

Kolom `species_encoded` kini merepresentasikan label dalam bentuk angka.

---

## Seleksi Fitur

---

## 3.7 Pemisahan Fitur dan Target

Dataset dipisahkan menjadi dua bagian agar model dapat dilatih secara *supervised*:
- **X** → matriks fitur input (4 kolom numerik)
- **y** → vektor target/label (1 kolom encoded)

```python
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species_encoded']
X.head()
```

**Output X — Fitur Input (5 baris pertama):**

| | sepal_length | sepal_width | petal_length | petal_width |
|--|---|---|---|---|
| **0** | 5.1 | 3.5 | 1.4 | 0.2 |
| **1** | 4.9 | 3.0 | 1.4 | 0.2 |
| **2** | 4.7 | 3.2 | 1.3 | 0.2 |
| **3** | 4.6 | 3.1 | 1.5 | 0.2 |
| **4** | 5.0 | 3.6 | 1.4 | 0.2 |

`y` = `[0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2]` (target klasifikasi, 50 sampel per kelas).

---

## Standardisasi Scaling

---

## 3.8 Alasan Dilakukan Scaling

Scaling penting untuk algoritma berbasis jarak seperti **KNN**, **K-Means**, dan **SVM** karena fitur dengan rentang nilai lebih besar dapat mendominasi perhitungan jarak dan membuat fitur lain tidak berpengaruh.

**Contoh masalah tanpa scaling:**

| Fitur | Range | Tanpa Scaling — Dominasi Jarak |
|-------|:-----:|-------------------------------|
| `sepal_length` | 4.3 – 7.9 cm | Rentang ≈ 3.6 |
| `petal_length` | 1.0 – 6.9 cm | Rentang ≈ 5.9 → **mendominasi** |
| `petal_width` | 0.1 – 2.5 cm | Rentang kecil → **terabaikan** |

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pd.DataFrame(X_scaled, columns=X.columns).head()
```

**Output Data Setelah Scaling (5 baris pertama):**

| | sepal_length | sepal_width | petal_length | petal_width |
|--|---|---|---|---|
| **0** | -0.9155 | 1.0199 | -1.3577 | -1.3359 |
| **1** | -1.1576 | -0.1280 | -1.3577 | -1.3359 |
| **2** | -1.3996 | 0.3311 | -1.4147 | -1.3359 |
| **3** | -1.5206 | 0.1015 | -1.3006 | -1.3359 |
| **4** | -1.0365 | 1.2495 | -1.3577 | -1.3359 |

Setelah scaling, seluruh fitur memiliki **mean ≈ 0** dan **standar deviasi ≈ 1**, sehingga tidak ada fitur yang mendominasi.

---

## Visualisasi Sebelum dan Sesudah Scaling

---

## 3.9 Sebelum Scaling

```python
X.hist(figsize=(8, 6))
plt.tight_layout()
plt.show()
```

![Distribusi Fitur Sebelum Scaling](Assets/Pertemuan3/SebelumScalling.png)

Histogram menunjukkan bahwa setiap fitur memiliki skala dan rentang yang berbeda-beda — `petal_length` memiliki rentang paling lebar.

---

## 3.10 Sesudah Scaling

```python
pd.DataFrame(X_scaled, columns=X.columns).hist(figsize=(8, 6))
plt.tight_layout()
plt.show()
```

![Distribusi Fitur Sesudah Scaling](Assets/Pertemuan3/SesudahScalling.png)

Setelah scaling, semua fitur berada pada skala yang sama (terpusat di 0), sehingga kontribusi setiap fitur terhadap perhitungan jarak menjadi seimbang.

---

## Mengukur Jarak (Distance)

---

## 3.13 Cara Mengukur Jarak untuk Data Iris

Karena seluruh fitur Iris bertipe numerik, terdapat beberapa metrik jarak yang dapat digunakan. **Scaling wajib dilakukan** sebelum menghitung jarak.

**Perbandingan Metrik Jarak Numerik:**

| Metrik | Formula | Parameter | Kapan Dipakai |
|--------|---------|:---------:|---------------|
| **Euclidean** | $d = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$ | — | Jarak garis lurus, data normal, paling umum |
| **Manhattan** | $d = \sum_{i=1}^{n}\|x_i - y_i\|$ | — | Lebih tahan outlier, cocok untuk data grid |
| **Minkowski** | $d = \left(\sum_{i=1}^{n}\|x_i - y_i\|^p\right)^{1/p}$ | p=1→Manhattan, p=2→Euclidean | Generalisasi keduanya, fleksibel |

### 3.13.1 Scaling Data

```python
X = df[numeric_cols].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 3.13.2 Distance Matrix — Euclidean

```python
D_euclid = pairwise_distances(X_scaled, metric='euclidean')
print("Euclidean D[0:5, 0:5]:\n", D_euclid[:5, :5].round(4))
```

### 3.13.3 Distance Matrix — Manhattan

```python
D_manhattan = pairwise_distances(X_scaled, metric='manhattan')
print("Manhattan D[0:5, 0:5]:\n", D_manhattan[:5, :5].round(4))
```

### 3.13.4 Distance Matrix — Minkowski (p=3)

```python
D_minkowski = pairwise_distances(X_scaled, metric='minkowski', p=3)
print("Minkowski(p=3) D[0:5, 0:5]:\n", D_minkowski[:5, :5].round(4))
```

**Perbandingan Nilai Jarak antara Iris-0 dan Iris-50 (setosa vs versicolor) setelah scaling:**

| Metrik | Nilai Jarak | Interpretasi |
|--------|:-----------:|-------------|
| Euclidean | ≈ 6.50 | Jarak garis lurus di ruang 4D |
| Manhattan | ≈ 10.20 | Jumlah selisih absolut per dimensi |
| Minkowski (p=3) | ≈ 5.40 | Lebih kecil dari Euclidean, sensifit ke outlier berbeda |

---

## Distance Matrix di Orange

---

## 3.14 Distance Matrix di Orange (Workflow)

Orange menyediakan widget **Distances** yang langsung menghitung distance matrix tanpa perlu menulis kode. Berikut langkah-langkahnya:

| Langkah | Widget | Keterangan |
|:-------:|--------|------------|
| 1 | **File** / **SQL Table** | Load dataset Iris |
| 2 | **Select Columns** | Masukkan `sepal_*`, `petal_*` ke Attributes; `species` ke Class |
| 3 | **Normalize** *(opsional)* | Pilih Standardize agar skala seragam |
| 4 | **Distances** | Pilih metric: Euclidean / Manhattan / Cosine |
| 5 | **Distance Matrix** | Tampilkan matriks jarak antar semua sampel |
| 6 | **Heat Map** / **Hierarchical Clustering** | Visualisasi pola jarak dan pengelompokan |

**Alur widget Orange (teks):**
```
[File] → [Select Columns] → [Normalize] → [Distances] → [Distance Matrix]
                                                       ↘ [Heat Map]
                                                       ↘ [Hierarchical Clustering]
```

![Workflow Orange — Pengukuran Jarak Iris di Orange](Assets/Pertemuan3/DataIrisOrangePengukuranJarak.png)

> **Gambar:** Workflow Orange yang menghitung 4 metrik jarak (Euclidean, Manhattan, Spearman, Hamming) dari data Iris yang dimuat melalui CSV File Import dan SQL Table, masing-masing diteruskan ke Distance Matrix dan disimpan via Save Distance Matrix.

---

## Jarak Data Campuran (Mixed-Type)

---

## 3.15 Pengukuran Jarak pada Data Campuran — Bank Marketing

Dataset **Bank Marketing** dipilih sebagai data campuran (*mixed-type*) untuk tugas ini karena mengandung **keempat tipe data sekaligus**: numerik, nominal/kategorikal, ordinal, dan biner. Dataset diperoleh dari dua sumber sekaligus: file CSV lokal (`bank.csv`) dan tabel PostgreSQL (`bank_marketing` di database `BankMarketing`).

### 3.15.1 Profil Dataset Bank Marketing

Dataset berisi data kampanye pemasaran bank Portugal dengan **4521 baris** dan **17 kolom**.

```python
df_bank = pd.read_csv("DataCampuranPertemuan3/Bank Marketing/bank.csv")
df_bank.head()
```

**Sampel 5 baris pertama:**

| age | job | marital | education | default | balance | housing | loan | contact | day | month | duration | campaign | pdays | previous | poutcome | deposit |
|:---:|-----|---------|-----------|:-------:|:-------:|:-------:|:----:|---------|:---:|-------|:--------:|:--------:|:-----:|:--------:|---------|:-------:|
| 59 | admin. | married | secondary | no | 2343 | yes | no | unknown | 5 | may | 1042 | 1 | -1 | 0 | unknown | yes |
| 56 | admin. | married | secondary | no | 45 | no | no | unknown | 5 | may | 1467 | 1 | -1 | 0 | unknown | yes |
| 41 | technician | married | secondary | no | 1270 | yes | no | unknown | 5 | may | 1389 | 1 | -1 | 0 | unknown | yes |
| 55 | services | married | secondary | no | 2476 | yes | no | unknown | 5 | may | 579 | 1 | -1 | 0 | unknown | yes |

### 3.15.2 Identifikasi Tipe Data per Kolom (Mixed-Type)

Seluruh kolom dataset dikelompokkan sesuai tipe datanya untuk menentukan metrik jarak yang sesuai:

| Kolom | Tipe Data | Nilai / Range | Metrik Jarak yang Sesuai |
|-------|-----------|---------------|:------------------------:|
| `age` | **Numerik** (int) | 18 – 95 tahun | Euclidean / Manhattan |
| `balance` | **Numerik** (int) | -∞ – ∞ (saldo rekening) | Euclidean / Manhattan |
| `duration` | **Numerik** (int) | detik durasi panggilan | Euclidean / Manhattan |
| `campaign` | **Numerik** (int) | jumlah kontak kampanye | Euclidean / Manhattan |
| `pdays` | **Numerik** (int) | -1 = belum pernah dihubungi | Euclidean / Manhattan |
| `previous` | **Numerik** (int) | kontak sebelumnya | Euclidean / Manhattan |
| `day` | **Numerik** (int) | hari kontak (1–31) | Euclidean / Manhattan |
| `education` | **Ordinal** | primary < secondary < tertiary | Spearman |
| `month` | **Ordinal** | jan < feb < … < dec | Spearman |
| `job` | **Nominal/Kategorikal** | 12 kategori pekerjaan | Hamming |
| `marital` | **Nominal** | married, single, divorced | Hamming |
| `contact` | **Nominal** | unknown, telephone, cellular | Hamming |
| `poutcome` | **Nominal** | unknown, failure, other, success | Hamming |
| `default` | **Biner** | yes / no | Hamming |
| `housing` | **Biner** | yes / no (kredit rumah) | Hamming |
| `loan` | **Biner** | yes / no (pinjaman pribadi) | Hamming |
| `deposit` | **Biner** | yes / no | **Target / Kelas** |

> **Kesimpulan:** Dataset Bank Marketing adalah contoh sempurna data campuran — terdapat fitur numerik kontinu, fitur ordinal berurutan, fitur nominal tanpa urutan, dan fitur biner, sehingga tidak ada satu metrik jarak tunggal yang cukup.

### 3.15.3 Mengapa Data Campuran Memerlukan Beberapa Metrik?

Setiap tipe data memiliki cara pengukuran jarak yang berbeda:

| Tipe Data | Contoh Kolom | Masalah Jika Salah Metrik | Solusi |
|-----------|-------------|--------------------------|--------|
| **Numerik** | `age`, `balance` | Tanpa normalisasi, `balance` mendominasi jarak | Euclidean/Manhattan setelah scaling |
| **Ordinal** | `education`, `month` | Nilai teks tidak bisa dijumlah langsung | Konversi ke rank → Spearman |
| **Nominal** | `job`, `marital` | "admin. vs technician" bukan selisih angka | Hamming (match/mismatch) |
| **Biner** | `housing`, `loan` | Cukup cek kesamaan nilai 0/1 | Hamming |

### 3.15.4 Koneksi ke Database PostgreSQL

Data Bank Marketing juga dimuat langsung dari database PostgreSQL menggunakan widget **SQL Table** di Orange:

| Parameter | Nilai |
|-----------|-------|
| **Server** | PostgreSQL |
| **Host** | `127.0.0.1` |
| **Database** | `BankMarketing` |
| **User** | `postgres` |
| **Table** | `bank_marketing` |
| **Total baris** | 239 (subset) |

![Koneksi SQL Table ke PostgreSQL BankMarketing](DataCampuranPertemuan3/Bank%20Marketing/Sudah%20Menghubungkan%20ke%20Postgree.png)

> **Gambar:** Widget SQL Table Orange berhasil terhubung ke database PostgreSQL `BankMarketing` dan memuat tabel `bank_marketing`. Tombol Connect berhasil, dan data tersedia untuk dialirkan ke pipeline distance measurement.

### 3.15.5 Workflow Orange — Pengukuran Jarak pada Data Campuran

Orange digunakan untuk mengukur jarak menggunakan **4 metrik berbeda** yang masing-masing sesuai dengan tipe data tertentu dalam Bank Marketing. Workflow yang dibangun:

```
[CSV File Import] ──Data──▶ [Data Table] ──Selected Data──▶ [Euclidean Distances] ──▶ [Distance Matrix Euclidean] ──▶ [Save]
  (bank.csv)                                               ──Selected Data──▶ [Manhattan Distances] ──▶ [Distance Matrix Manhattan] ──▶ [Save]
                                                           ──Selected Data──▶ [Spearman Distances]  ──▶ [Distance Matrix Spearman]  ──▶ [Save]
                                                           ──Selected Data──▶ [Hamming Distances]   ──▶ [Distance Matrix Hamming]   ──▶ [Save]

[SQL Table] ──────Data──▶ [Data Table (1)] ──Same 4 distance pipelines──▶ ...
  (BankMarketing DB)
```

**Penjelasan 4 Metrik yang Dipakai:**

| Metrik | Cocok Untuk | Cara Kerja |
|--------|-------------|-----------|
| **Euclidean** | Fitur numerik | $d = \sqrt{\sum(x_i - y_i)^2}$ — jarak garis lurus di ruang fitur numerik |
| **Manhattan** | Fitur numerik (robust outlier) | $d = \sum\|x_i - y_i\|$ — jumlah selisih absolut |
| **Spearman** | Fitur ordinal | Menghitung korelasi rank antar baris; cocok untuk `education`, `month` yang memiliki urutan |
| **Hamming** | Fitur kategorikal & biner | Menghitung proporsi posisi yang berbeda; cocok untuk `job`, `marital`, `housing`, `loan`, `default` |

```{admonition} Mengapa 4 Metrik Sekaligus?
:class: tip
Karena data Bank Marketing bersifat **mixed-type**, tidak ada satu metrik yang sempurna untuk semua kolom. Dengan menjalankan 4 metrik:
- **Euclidean & Manhattan** mengukur jarak berdasarkan fitur numerik (usia, saldo, durasi).
- **Spearman** sensitif terhadap peringkat — ideal untuk `education` (primary < secondary < tertiary) dan `month`.
- **Hamming** menghitung perbedaan kategorikal — ideal untuk `job`, `marital`, `contact`, `default`, `housing`, `loan`.
```

![Workflow Orange — Pengukuran Jarak Data Campuran Bank Marketing](DataCampuranPertemuan3/Bank%20Marketing/MelakukanDistancediOrange.png)

> **Gambar:** Workflow lengkap `DataCampuranBank.ows` di Orange Data Mining. Terdapat dua sumber data: **CSV File Import** (atas) dan **SQL Table** / database PostgreSQL (bawah), masing-masing dialirkan ke **Data Table** lalu ke empat widget **Distance** (Euclidean, Manhattan, Spearman, Hamming) → **Distance Matrix** → **Save Distance Matrix**. Ikon biru menandakan semua proses berhasil diselesaikan.

### 3.15.6 Download File Orange Workflow

File workflow Orange yang digunakan untuk mengukur jarak pada data Bank Marketing dapat diunduh di bawah ini:

```{admonition} 📥 Download File
:class: note
{download}`DataCampuranBank.ows — Workflow Pengukuran Jarak Mixed-Type <DataCampuranPertemuan3/Bank Marketing/DataCampuranBank.ows>`

File ini berisi seluruh pipeline Orange: dari loading data CSV/SQL, hingga perhitungan Euclidean, Manhattan, Spearman, dan Hamming Distance Matrix.
```

### 3.15.7 Konsep Gower Distance (Referensi Teoritis)

Untuk pengukuran jarak data campuran secara teori matematis, digunakan **Gower Distance** yang menggabungkan semua tipe data dengan formula:

$$d_{Gower}(x, y) = \frac{1}{p}\sum_{i=1}^{p} d_i(x_i, y_i)$$

| Tipe Fitur | Cara Hitung Komponen $d_i$ | Formula |
|-----------|--------------------------|---------|
| **Numerik** | Selisih dinormalisasi dengan range | $\frac{\|x_i - y_i\|}{range_i}$ |
| **Nominal** | Sama = 0, Beda = 1 | $0$ jika $x_i = y_i$, else $1$ |
| **Biner** | Sama = 0, Beda = 1 | $0$ jika $x_i = y_i$, else $1$ |
| **Ordinal** | Selisih posisi dinormalisasi | $\frac{\|rank(x_i) - rank(y_i)\|}{k-1}$ |

Dalam praktik Orange, Gower Distance diimplementasikan secara terpisah per tipe menggunakan metrik Euclidean (numerik), Spearman (ordinal), dan Hamming (nominal/biner) — seperti yang telah dilakukan dalam tugas ini.

---

## Menyimpan Dataset Final untuk Modeling

---

## 3.16 Menyimpan Dataset Final

Setelah seluruh proses preparation selesai, dataset yang sudah di-scale disimpan sebagai file CSV baru untuk digunakan pada tahap **Modeling**.

```python
df_modeling = pd.DataFrame(X_scaled, columns=X.columns)
df_modeling['target'] = y.values
df_modeling.to_csv("IRIS_after_preparation_for_modeling.csv", index=False)
df_modeling.head()
```

**Output `df_modeling.head()` — Dataset Siap Modeling:**

| | sepal_length | sepal_width | petal_length | petal_width | target |
|--|---|---|---|---|:---:|
| **0** | -0.9155 | 1.0199 | -1.3577 | -1.3359 | 0.0 |
| **1** | -1.1576 | -0.1280 | -1.3577 | -1.3359 | 0.0 |
| **2** | -1.3996 | 0.3311 | -1.4147 | -1.3359 | 0.0 |
| **3** | -1.5206 | 0.1015 | -1.3006 | -1.3359 | 0.0 |
| **4** | -1.0365 | 1.2495 | -1.3577 | -1.3359 | 0.0 |

Dataset ini telah siap digunakan untuk tahap Modeling (KNN, Decision Tree, SVM, dll). Kolom `target` berisi label encoded (0 = setosa, 1 = versicolor, 2 = virginica).

---

```{admonition} Identitas Mahasiswa
:class: note

**Nama:** Zakaria Mujur Prasetyo | **NIM:** 240411100144
```
