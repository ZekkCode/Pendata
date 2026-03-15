# Lampiran Materi Pertemuan 3.1 — Bukti Gambar dan Identifikasi Missing Value

## Daftar Isi
```{dropdown} Klik untuk membuka Daftar Isi
:open:

1. [Bukti Gambar Dataset dan Workflow Orange](#1-bukti-gambar-dataset-dan-workflow-orange)
2. [Identifikasi Missing Value](#2-identifikasi-missing-value)
3. [Contoh Kode Pembuatan Missing Value Buatan](#3-contoh-kode-pembuatan-missing-value-buatan)
4. [Visualisasi Hasil Pemeriksaan](#4-visualisasi-hasil-pemeriksaan)
```

Dokumen ini merupakan **lampiran dari Pertemuan 3.1** yang berisi bukti gambar, kode untuk membuat missing value buatan, serta langkah identifikasi data di **Orange/OWS** dan **Python**.

---

## 1. Bukti Gambar Dataset dan Workflow Orange

### 1.1 Workflow Pengukuran Jarak Dataset Iris di Orange

![Workflow pengukuran jarak Iris di Orange](Assets/Pertemuan3/DataIrisOrangePengukuranJarak.png)

Workflow Orange untuk dataset Iris dimulai dari **CSV File Import**, dilanjutkan ke **Data Table**, kemudian ke widget perhitungan jarak (**Euclidean Distances**, **Manhattan Distances**, **Spearman Distances**, **Hamming Distances**), lalu ditampilkan pada **Distance Matrix** dan disimpan melalui **Save Distance Matrix**.

### 1.2 Bukti Data CSV ke PostgreSQL

![Bukti data CSV telah dimasukkan ke PostgreSQL](Assets/Pertemuan3/Gambar%20Csv%20ke%20PostgreeSQL.png)

Dataset Iris telah berhasil dimasukkan ke PostgreSQL dan dapat diakses melalui query `SELECT * FROM public.iris`. Total data: 150 baris.

### 1.3 Histogram Fitur Iris — Sebelum Scaling

![Histogram sebelum scaling](Assets/Pertemuan3/SebelumScalling.png)

Histogram di atas menunjukkan distribusi keempat fitur numerik (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) **sebelum** dilakukan scaling. Terlihat bahwa skala antar fitur berbeda-beda.

### 1.4 Histogram Fitur Iris — Sesudah Scaling

![Histogram sesudah scaling](Assets/Pertemuan3/SesudahScalling.png)

Setelah dilakukan **StandardScaler**, distribusi keempat fitur sudah dinormalisasi dengan mean ≈ 0 dan std ≈ 1, sehingga skala antar fitur menjadi setara untuk perhitungan jarak.

---

## 2. Identifikasi Missing Value

### 2.1 Pemeriksaan Missing Value Awal

Berdasarkan hasil pemeriksaan awal, **dataset Iris tidak memiliki missing value**. Begitu pula **dataset Bank** tidak memiliki missing value.

Oleh karena itu, pada tahap berikutnya dibuat **satu missing value buatan** untuk keperluan simulasi imputasi KNN.

### 2.2 Langkah Identifikasi Missing Value di Orange

1. Buka **Orange**
2. Tambahkan widget **CSV File Import** untuk data file `.csv` atau **SQL Table** untuk data dari database
3. Hubungkan ke widget **Data Table**
4. Amati isi tabel untuk melihat apakah ada sel kosong
5. Tambahkan widget **Data Info** untuk melihat ringkasan jumlah instance dan fitur
6. Catat apakah dataset memiliki missing value atau tidak

### 2.3 Identifikasi pada File Workflow `.ows`

1. Buka file workflow `.ows` (misalnya `DataCampuranBank.ows`)
2. Periksa widget input data (**CSV File Import** atau **SQL Table**)
3. Pastikan data terhubung ke **Data Table**
4. Klik **Data Table** untuk melihat isi data
5. Periksa apakah ada nilai kosong pada kolom tertentu

---

## 3. Kode: Membuat Missing Value Buatan dan Gambar Bukti — Dataset Iris

Kode berikut dijalankan di **Jupyter Notebook** atau **Google Colab**.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== LOAD DATASET =====
iris = pd.read_csv('IRIS.csv')

# ===== CEK MISSING VALUE AWAL =====
print('Missing value awal per kolom:')
print(iris.isnull().sum())
print()

# ===== BUAT MISSING VALUE BUATAN =====
row_idx = 5
col_name = 'petal_width'
nilai_asli_iris = iris.loc[row_idx, col_name]
print(f'Nilai asli pada baris {row_idx}, kolom {col_name}: {nilai_asli_iris}')

iris_missing = iris.copy()
iris_missing.loc[row_idx, col_name] = np.nan

print('\nData baris 5 setelah dibuat missing value:')
print(iris_missing.loc[[row_idx]])
```

**Output yang diharapkan:**

```
Missing value awal per kolom:
sepal_length    0
sepal_width     0
petal_length    0
petal_width     0
species         0
dtype: int64

Nilai asli pada baris 5, kolom petal_width: 0.4

Data baris 5 setelah dibuat missing value:
   sepal_length  sepal_width  petal_length  petal_width      species
5           5.4          3.9           1.7          NaN  Iris-setosa
```

### 3.1 Gambar Grafik Cek Missing Value Iris

```python
# Grafik jumlah missing value per kolom
missing_count = iris.isnull().sum()

plt.figure(figsize=(8, 4))
missing_count.plot(kind='bar', color='steelblue')
plt.title('Jumlah Missing Value per Kolom - Iris (Sebelum Dibuat Buatan)')
plt.xlabel('Kolom')
plt.ylabel('Jumlah Missing')
plt.tight_layout()
plt.savefig('cek-missing-value-iris.png', dpi=200, bbox_inches='tight')
plt.show()
```

### 3.2 Gambar Tabel Missing Value Buatan Iris

```python
# Tabel baris yang memiliki missing value
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('off')
tabel = ax.table(
    cellText=iris_missing.loc[[row_idx]].values,
    colLabels=iris_missing.columns,
    loc='center'
)
tabel.auto_set_font_size(False)
tabel.set_fontsize(9)
tabel.scale(1, 1.5)
plt.title('Baris 5 dengan Missing Value Buatan (petal_width = NaN)', fontsize=11)
plt.tight_layout()
plt.savefig('missing-iris-baris5.png', dpi=200, bbox_inches='tight')
plt.show()
```

---

## 4. Kode: Membuat Missing Value Buatan dan Gambar Bukti — Dataset Bank

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== LOAD DATASET =====
bank = pd.read_csv('bank.csv')

# ===== CEK MISSING VALUE AWAL =====
print('Missing value awal per kolom:')
print(bank.isnull().sum())
print()

# ===== BUAT MISSING VALUE BUATAN =====
row_idx = 5
col_name = 'balance'
nilai_asli_bank = bank.loc[row_idx, col_name]
print(f'Nilai asli pada baris {row_idx}, kolom {col_name}: {nilai_asli_bank}')

bank_missing = bank.copy()
bank_missing.loc[row_idx, col_name] = np.nan

print('\nData baris 5 setelah dibuat missing value:')
print(bank_missing.loc[[row_idx]])
```

**Output yang diharapkan:**

```
Missing value awal per kolom:
age          0
job          0
marital      0
education    0
default      0
balance      0
housing      0
loan         0
contact      0
day          0
month        0
duration     0
campaign     0
pdays        0
previous     0
poutcome     0
deposit      0
dtype: int64

Nilai asli pada baris 5, kolom balance: 0

Data baris 5 setelah dibuat missing value:
   age          job marital education default  balance housing loan  ...
5   42  management  single  tertiary      no      NaN     yes  yes  ...
```

### 4.1 Gambar Grafik Cek Missing Value Bank

```python
missing_count_bank = bank.isnull().sum()

plt.figure(figsize=(12, 4))
missing_count_bank.plot(kind='bar', color='steelblue')
plt.title('Jumlah Missing Value per Kolom - Bank (Sebelum Dibuat Buatan)')
plt.xlabel('Kolom')
plt.ylabel('Jumlah Missing')
plt.tight_layout()
plt.savefig('cek-missing-value-bank.png', dpi=200, bbox_inches='tight')
plt.show()
```

### 4.2 Gambar Tabel Missing Value Buatan Bank

```python
kolom_tampil = ['age', 'job', 'marital', 'education', 'balance', 'housing', 'loan', 'deposit']
kolom_tampil = [k for k in kolom_tampil if k in bank_missing.columns]

fig, ax = plt.subplots(figsize=(14, 2))
ax.axis('off')
tabel = ax.table(
    cellText=bank_missing.loc[[row_idx], kolom_tampil].values,
    colLabels=kolom_tampil,
    loc='center'
)
tabel.auto_set_font_size(False)
tabel.set_fontsize(9)
tabel.scale(1, 1.5)
plt.title('Baris 5 dengan Missing Value Buatan (balance = NaN)', fontsize=11)
plt.tight_layout()
plt.savefig('missing-bank-baris5.png', dpi=200, bbox_inches='tight')
plt.show()
```

---

## 5. Kode: Perhitungan Jarak dan Imputasi KNN — Dataset Iris

```python
import pandas as pd
import numpy as np

# Load dan buat missing value
iris = pd.read_csv("IRIS.csv")
target_idx = 5
col_missing = 'petal_width'
nilai_asli = iris.loc[target_idx, col_missing]

iris_knn = iris.copy()
iris_knn.loc[target_idx, col_missing] = np.nan

# Hitung jarak Euclidean (tanpa kolom yang missing)
fitur = ['sepal_length', 'sepal_width', 'petal_length']
hasil_jarak = []

for i in range(len(iris_knn)):
    if i == target_idx:
        continue
    d = np.sqrt(((iris_knn.loc[target_idx, fitur] - iris_knn.loc[i, fitur]) ** 2).sum())
    hasil_jarak.append((i, d, iris.loc[i, col_missing]))

hasil_jarak = sorted(hasil_jarak, key=lambda x: x[1])

# Ambil 3 tetangga terdekat
k = 3
tetangga = hasil_jarak[:k]
imputasi = np.mean([x[2] for x in tetangga])

print("=" * 50)
print("IMPUTASI KNN - DATASET IRIS")
print("=" * 50)
print(f"Baris target: {target_idx}")
print(f"Kolom missing: {col_missing}")
print(f"Nilai asli: {nilai_asli}")
print(f"\n3 tetangga terdekat:")
for t in tetangga:
    data_row = iris.loc[t[0], fitur].values
    print(f"  Baris {t[0]}: jarak = {t[1]:.4f}, {col_missing} = {t[2]}, data = {data_row}")
print(f"\nHasil imputasi (rata-rata): {imputasi:.4f}")
```

**Output:**

```
==================================================
IMPUTASI KNN - DATASET IRIS
==================================================
Baris target: 5
Kolom missing: petal_width
Nilai asli: 0.4

3 tetangga terdekat:
  Baris 10: jarak = 0.2828, petal_width = 0.2, data = [5.4 3.7 1.5]
  Baris 48: jarak = 0.3000, petal_width = 0.2, data = [5.3 3.7 1.5]
  Baris 18: jarak = 0.3162, petal_width = 0.3, data = [5.7 3.8 1.7]

Hasil imputasi (rata-rata): 0.2333
```

---

## 6. Kode: Perhitungan Jarak dan Imputasi KNN — Dataset Bank (Data Campuran)

```python
import pandas as pd
import numpy as np

# Load dan buat missing value
bank = pd.read_csv("bank.csv")
target_idx = 5
col_missing = 'balance'
nilai_asli = bank.loc[target_idx, col_missing]

bank_knn = bank.copy()
bank_knn.loc[target_idx, col_missing] = np.nan

# Konversi ordinal
edu_order = {'primary': 1, 'secondary': 2, 'tertiary': 3}
m_edu = 3

def ord_norm(val):
    return (edu_order[val] - 1) / (m_edu - 1)

# Filter education valid
bank_knn = bank_knn[bank_knn['education'].isin(edu_order.keys())].copy()
bank_knn = bank_knn.reset_index(drop=True)
missing_idx = bank_knn[bank_knn[col_missing].isna()].index[0]

# Normalisasi age (Min-Max)
age_min = bank_knn['age'].min()
age_max = bank_knn['age'].max()

def age_norm(val):
    return (val - age_min) / (age_max - age_min)

# Hitung jarak campuran
cat_cols = ['marital', 'housing', 'loan']
hasil_jarak = []

for i in range(len(bank_knn)):
    if i == missing_idx:
        continue

    # Numerik + Ordinal (Euclidean)
    d_numord = np.sqrt(
        (age_norm(bank_knn.loc[missing_idx, 'age']) - age_norm(bank_knn.loc[i, 'age']))**2 +
        (ord_norm(bank_knn.loc[missing_idx, 'education']) - ord_norm(bank_knn.loc[i, 'education']))**2
    )

    # Kategorikal
    P = len(cat_cols)
    M = sum(bank_knn.loc[missing_idx, c] == bank_knn.loc[i, c] for c in cat_cols)
    d_cat = (P - M) / P

    d_total = d_numord + d_cat

    bal = bank_knn.loc[i, col_missing]
    if pd.isna(bal):
        continue
    hasil_jarak.append((i, d_total, d_numord, d_cat, bal))

hasil_jarak = sorted(hasil_jarak, key=lambda x: x[1])

# 3 tetangga terdekat
k = 3
tetangga = hasil_jarak[:k]
imputasi = np.mean([x[4] for x in tetangga])

print("=" * 60)
print("IMPUTASI KNN - DATASET BANK (DATA CAMPURAN)")
print("=" * 60)
print(f"Baris target: {missing_idx}")
print(f"Kolom missing: {col_missing}")
print(f"Nilai asli: {nilai_asli}")
print(f"age_min={age_min}, age_max={age_max}")
print(f"\n3 tetangga terdekat:")
for t in tetangga:
    r = bank_knn.loc[t[0]]
    print(f"  Baris {t[0]}: d_total={t[1]:.4f} (d_numord={t[2]:.4f}, d_cat={t[3]:.4f}), balance={t[4]}")
    print(f"    age={r['age']}, edu={r['education']}, mar={r['marital']}, hou={r['housing']}, loan={r['loan']}")
print(f"\nHasil imputasi (rata-rata): {imputasi:.2f}")
```

**Output:**

```
============================================================
IMPUTASI KNN - DATASET BANK (DATA CAMPURAN)
============================================================
Baris target: 5
Kolom missing: balance
Nilai asli: 0
age_min=23, age_max=60

3 tetangga terdekat:
  Baris 96: d_total=0.3243 (d_numord=0.3243, d_cat=0.0000), balance=880
    age=30, edu=tertiary, mar=single, hou=yes, loan=yes
  Baris 21: d_total=0.3604 (d_numord=0.0270, d_cat=0.3333), balance=2067
    age=43, edu=tertiary, mar=single, hou=yes, loan=no
  Baris 51: d_total=0.4144 (d_numord=0.0811, d_cat=0.3333), balance=517
    age=39, edu=tertiary, mar=divorced, hou=yes, loan=yes

Hasil imputasi (rata-rata): 1154.67
```

---

## 7. Kode: Membuat Gambar Tabel Hasil Imputasi

### 7.1 Gambar Hasil Imputasi Iris

```python
import matplotlib.pyplot as plt

# Setelah imputasi
iris_hasil = iris_knn.copy()
iris_hasil.loc[target_idx, 'petal_width'] = 0.2333  # hasil imputasi

fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('off')
tabel = ax.table(
    cellText=iris_hasil.loc[[target_idx]].values,
    colLabels=iris_hasil.columns,
    loc='center'
)
tabel.auto_set_font_size(False)
tabel.set_fontsize(9)
tabel.scale(1, 1.5)
plt.title('Baris 5 Setelah Imputasi KNN (petal_width = 0.2333)', fontsize=11)
plt.tight_layout()
plt.savefig('hasil-imputasi-iris.png', dpi=200, bbox_inches='tight')
plt.show()
```

### 7.2 Gambar Hasil Imputasi Bank

```python
kolom_tampil = ['age', 'job', 'marital', 'education', 'balance', 'housing', 'loan', 'deposit']
kolom_tampil = [k for k in kolom_tampil if k in bank_knn.columns]

bank_hasil = bank_knn.copy()
bank_hasil.loc[missing_idx, 'balance'] = 1154.67  # hasil imputasi

fig, ax = plt.subplots(figsize=(14, 2))
ax.axis('off')
tabel = ax.table(
    cellText=bank_hasil.loc[[missing_idx], kolom_tampil].values,
    colLabels=kolom_tampil,
    loc='center'
)
tabel.auto_set_font_size(False)
tabel.set_fontsize(9)
tabel.scale(1, 1.5)
plt.title('Baris 5 Setelah Imputasi KNN (balance = 1154.67)', fontsize=11)
plt.tight_layout()
plt.savefig('hasil-imputasi-bank.png', dpi=200, bbox_inches='tight')
plt.show()
```

---

## 8. Kode: Tabel Perbandingan Jarak 10 Teratas

### 8.1 Tabel Jarak Iris

```python
import matplotlib.pyplot as plt

# Data 10 teratas dari perhitungan sebelumnya
data_tabel = []
for t in hasil_jarak[:10]:
    data_tabel.append([t[0], f"{t[1]:.4f}", t[2]])

fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')
tabel = ax.table(
    cellText=data_tabel,
    colLabels=['Baris', 'Jarak', 'petal_width'],
    loc='center'
)
tabel.auto_set_font_size(False)
tabel.set_fontsize(9)
tabel.scale(1, 1.5)

# Warnai 3 baris teratas (tetangga terdekat)
for row in range(1, 4):
    for col in range(3):
        tabel[row, col].set_facecolor('#d4edda')

plt.title('10 Jarak Terkecil - Dataset Iris (3 hijau = tetangga terpilih)', fontsize=11)
plt.tight_layout()
plt.savefig('tabel-jarak-iris.png', dpi=200, bbox_inches='tight')
plt.show()
```

### 8.2 Tabel Jarak Bank

```python
data_tabel_bank = []
for t in hasil_jarak[:10]:
    r = bank_knn.loc[t[0]]
    data_tabel_bank.append([
        t[0],
        f"{t[2]:.4f}",
        f"{t[3]:.4f}",
        f"{t[1]:.4f}",
        t[4]
    ])

fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')
tabel = ax.table(
    cellText=data_tabel_bank,
    colLabels=['Baris', 'd_num+ord', 'd_kat', 'd_total', 'balance'],
    loc='center'
)
tabel.auto_set_font_size(False)
tabel.set_fontsize(9)
tabel.scale(1, 1.5)

for row in range(1, 4):
    for col in range(5):
        tabel[row, col].set_facecolor('#d4edda')

plt.title('10 Jarak Terkecil - Dataset Bank (3 hijau = tetangga terpilih)', fontsize=11)
plt.tight_layout()
plt.savefig('tabel-jarak-bank.png', dpi=200, bbox_inches='tight')
plt.show()
```

---

## 9. Daftar Semua Gambar yang Dihasilkan

Setelah menjalankan semua kode di atas, file gambar berikut akan tersimpan:

| No | Nama File | Keterangan |
|---|---|---|
| 1 | `cek-missing-value-iris.png` | Grafik cek missing value Iris (semua 0) |
| 2 | `missing-iris-baris5.png` | Tabel baris 5 Iris dengan NaN |
| 3 | `cek-missing-value-bank.png` | Grafik cek missing value Bank (semua 0) |
| 4 | `missing-bank-baris5.png` | Tabel baris 5 Bank dengan NaN |
| 5 | `hasil-imputasi-iris.png` | Tabel baris 5 Iris setelah imputasi |
| 6 | `hasil-imputasi-bank.png` | Tabel baris 5 Bank setelah imputasi |
| 7 | `tabel-jarak-iris.png` | Tabel 10 jarak terkecil Iris |
| 8 | `tabel-jarak-bank.png` | Tabel 10 jarak terkecil Bank |

Gambar dari folder `Assets/Pertemuan3/`:

| No | Nama File | Keterangan |
|---|---|---|
| 1 | `SebelumScalling.png` | Histogram Iris sebelum scaling |
| 2 | `SesudahScalling.png` | Histogram Iris sesudah scaling |
| 3 | `DataIrisOrangePengukuranJarak.png` | Workflow Orange pengukuran jarak |
| 4 | `Gambar Csv ke PostgreeSQL.png` | Bukti data di PostgreSQL |

---

## 10. Template Markdown untuk Laporan

Template berikut bisa langsung di-copy paste ke laporan:

```markdown
# Bukti Missing Value dan Hasil Imputasi KNN

## Dataset Iris

### Cek Missing Value Awal
![Cek missing value Iris](cek-missing-value-iris.png)
Semua kolom menunjukkan 0 missing value.

### Missing Value Buatan
![Missing value buatan Iris](missing-iris-baris5.png)
Kolom petal_width pada baris 5 dihilangkan untuk simulasi.

### Tabel Jarak Terkecil
![Tabel jarak Iris](tabel-jarak-iris.png)
3 tetangga terdekat: baris 10 (0.2828), baris 48 (0.3000), baris 18 (0.3162).

### Hasil Imputasi
![Hasil imputasi Iris](hasil-imputasi-iris.png)
petal_width diisi dengan rata-rata: (0.2 + 0.2 + 0.3) / 3 = **0.2333**

### Workflow Orange
![Workflow Orange Iris](Assets/Pertemuan3/DataIrisOrangePengukuranJarak.png)

---

## Dataset Bank (Data Campuran)

### Cek Missing Value Awal
![Cek missing value Bank](cek-missing-value-bank.png)
Semua kolom menunjukkan 0 missing value.

### Missing Value Buatan
![Missing value buatan Bank](missing-bank-baris5.png)
Kolom balance pada baris 5 dihilangkan untuk simulasi.

### Tabel Jarak Terkecil
![Tabel jarak Bank](tabel-jarak-bank.png)
3 tetangga terdekat: baris 96 (0.3243), baris 21 (0.3604), baris 51 (0.4144).

### Hasil Imputasi
![Hasil imputasi Bank](hasil-imputasi-bank.png)
balance diisi dengan rata-rata: (880 + 2067 + 517) / 3 = **1154.67**

### Bukti Data di PostgreSQL
![Data di PostgreSQL](Assets/Pertemuan3/Gambar%20Csv%20ke%20PostgreeSQL.png)
```
