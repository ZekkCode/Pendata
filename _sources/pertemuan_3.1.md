# Materi Pertemuan 3.1 — Imputasi Missing Value dengan KNN

## Daftar Isi
```{dropdown} Klik untuk membuka Daftar Isi
:open:

1. [Konsep KNN untuk Mengisi Missing Value](#1-konsep-knn-untuk-mengisi-missing-value)
2. [Aturan Menghitung Jarak Berdasarkan Tipe Data](#2-aturan-menghitung-jarak-berdasarkan-tipe-data)
3. [Contoh KNN pada Dataset Iris (Data Numerik)](#3-contoh-knn-pada-dataset-iris-data-numerik)
4. [Contoh KNN pada Dataset Bank (Data Campuran)](#4-contoh-knn-pada-dataset-bank-data-campuran)
5. [Verifikasi dengan Python](#5-verifikasi-dengan-python)
```

Dokumen ini merupakan **lanjutan dari Pertemuan 3 (Data Preparation)**, yang membahas secara lengkap cara mengisi missing value menggunakan metode **K-Nearest Neighbors (KNN)** pada dua jenis dataset:

1. **Dataset Iris** — data numerik murni
2. **Dataset Bank** — data campuran (numerik, ordinal, kategorikal)

Seluruh perhitungan dilakukan secara **manual langkah demi langkah**, kemudian diverifikasi dengan kode Python.

---

## 1. Konsep KNN untuk Mengisi Missing Value

### 1.1 Apa itu KNN Imputation?

KNN Imputation adalah metode untuk mengisi nilai yang hilang (missing value) dengan cara:

1. Mengukur **jarak** dari baris yang memiliki missing value ke semua baris lain
2. Memilih **k baris terdekat** (tetangga terdekat)
3. Mengisi missing value berdasarkan nilai dari tetangga terdekat tersebut

### 1.2 Aturan Pengisian

| Tipe Nilai yang Hilang | Cara Pengisian | Metode |
|---|---|---|
| **Numerik** | Rata-rata dari k tetangga terdekat | KNN Regression |
| **Kategorikal** | Modus (voting mayoritas) dari k tetangga | KNN Classification |

Pada pertemuan ini digunakan **KNN Regression** karena nilai yang diisi berupa angka.

### 1.3 Catatan Penting

- Jika satu kolom memiliki missing value, maka **kolom tersebut tidak diikutkan** dalam perhitungan jarak
- Contoh: jika kolom ke-4 kosong, maka jarak hanya dihitung dari kolom 1, 2, dan 3
- Setelah jarak dihitung, diambil **k data terdekat** (pada contoh ini k = 3)

---

## 2. Aturan Menghitung Jarak Berdasarkan Tipe Data

### 2.1 Jarak untuk Data Numerik — Euclidean Distance

Untuk data numerik digunakan **Euclidean Distance**:

$$
d(i,j) = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + \cdots + (x_p - y_p)^2}
$$

**Aturan:** Jika ada kolom yang missing, kolom tersebut **dilewati** dan tidak masuk perhitungan.

Contoh: Baris A = `[5.4, 3.9, 1.7, ?]` dan Baris B = `[5.1, 3.5, 1.4, 0.2]`

Karena kolom ke-4 kosong, jarak hanya dihitung dari 3 kolom pertama:

$$
d = \sqrt{(5.4-5.1)^2 + (3.9-3.5)^2 + (1.7-1.4)^2} = \sqrt{0.09 + 0.16 + 0.09} = \sqrt{0.34} = 0.5831
$$

---

### 2.2 Konversi Data Ordinal ke Numerik

Data ordinal memiliki urutan (ranking), tetapi tidak bisa langsung dimasukkan ke Euclidean. Harus diubah dulu menjadi numerik menggunakan rumus:

$$
z = \frac{r - 1}{m - 1}
$$

Keterangan:

- $r$ = urutan level ordinal (dimulai dari 1)
- $m$ = jumlah seluruh level

**Contoh:** Kolom `education` memiliki 3 level:

| Level | Urutan ($r$) | Hasil Konversi |
|---|---|---|
| primary | 1 | $(1-1)/(3-1) = 0$ |
| secondary | 2 | $(2-1)/(3-1) = 0.5$ |
| tertiary | 3 | $(3-1)/(3-1) = 1$ |

Setelah dikonversi, kolom ordinal **bisa digabungkan** dengan data numerik lain dan dihitung menggunakan Euclidean distance.

---

### 2.3 Jarak untuk Data Kategorikal

Untuk data kategorikal (nominal) digunakan rumus **ketidaksamaan**:

$$
d_{kat} = \frac{P - M}{P}
$$

Keterangan:

- $P$ = banyaknya kolom kategorikal yang dibandingkan
- $M$ = jumlah kolom yang **nilainya sama**

**Contoh:** Dibandingkan 3 kolom kategorikal (`marital`, `housing`, `loan`):

| Kolom | Baris A | Baris B | Sama? |
|---|---|---|---|
| marital | single | single | ✓ |
| housing | yes | yes | ✓ |
| loan | yes | no | ✗ |

Maka $P = 3$, $M = 2$ (dua kolom sama):

$$
d_{kat} = \frac{3 - 2}{3} = \frac{1}{3} = 0.3333
$$

---

### 2.4 Jarak Total untuk Data Campuran

Untuk data campuran, langkah-langkahnya adalah:

1. **Hitung jarak numerik + ordinal** → konversi ordinal ke numerik, lalu hitung Euclidean
2. **Hitung jarak kategorikal** → gunakan rumus $(P-M)/P$
3. **Jumlahkan keduanya** sebagai jarak total

$$
d_{total} = d_{num+ord} + d_{kat}
$$

Inilah inti dari perhitungan jarak pada data campuran: **ordinal dijadikan numerik → dihitung Euclidean, kategorikal dihitung sendiri → lalu keduanya dijumlahkan**.

---

## 3. Penerapan pada Dataset Iris (Data Numerik)

Dataset Iris seluruhnya numerik (kecuali kolom `species`), sehingga perhitungan jarak menggunakan **Euclidean distance** saja.

### 3.1 Cek Missing Value Awal

Dataset Iris asli **tidak memiliki missing value**, sehingga perlu dibuat missing value buatan untuk simulasi.

### 3.2 Membuat Missing Value Buatan

Nilai yang dihilangkan:

- **Baris indeks 5** (baris ke-6)
- **Kolom `petal_width`**

Data asli baris 5:

| sepal_length | sepal_width | petal_length | petal_width | species |
|---|---|---|---|---|
| 5.4 | 3.9 | 1.7 | **0.4** | Iris-setosa |

Setelah dihilangkan:

| sepal_length | sepal_width | petal_length | petal_width | species |
|---|---|---|---|---|
| 5.4 | 3.9 | 1.7 | **?** | Iris-setosa |

Karena `petal_width` kosong, maka jarak dihitung hanya dari 3 kolom: `sepal_length`, `sepal_width`, `petal_length`.

---

### 3.3 Perhitungan Jarak Manual (Euclidean)

Berikut perhitungan jarak dari **baris 5** ke beberapa baris lain:

#### Baris 5 vs Baris 0

Baris 0: `[5.1, 3.5, 1.4]`

$$
d(5,0) = \sqrt{(5.4-5.1)^2 + (3.9-3.5)^2 + (1.7-1.4)^2}
$$

$$
= \sqrt{0.3^2 + 0.4^2 + 0.3^2} = \sqrt{0.09 + 0.16 + 0.09} = \sqrt{0.34} = 0.5831
$$

#### Baris 5 vs Baris 1

Baris 1: `[4.9, 3.0, 1.4]`

$$
d(5,1) = \sqrt{(5.4-4.9)^2 + (3.9-3.0)^2 + (1.7-1.4)^2}
$$

$$
= \sqrt{0.5^2 + 0.9^2 + 0.3^2} = \sqrt{0.25 + 0.81 + 0.09} = \sqrt{1.15} = 1.0724
$$

#### Baris 5 vs Baris 2

Baris 2: `[4.7, 3.2, 1.3]`

$$
d(5,2) = \sqrt{(5.4-4.7)^2 + (3.9-3.2)^2 + (1.7-1.3)^2}
$$

$$
= \sqrt{0.7^2 + 0.7^2 + 0.4^2} = \sqrt{0.49 + 0.49 + 0.16} = \sqrt{1.14} = 1.0677
$$

#### Baris 5 vs Baris 3

Baris 3: `[4.6, 3.1, 1.5]`

$$
d(5,3) = \sqrt{(5.4-4.6)^2 + (3.9-3.1)^2 + (1.7-1.5)^2}
$$

$$
= \sqrt{0.8^2 + 0.8^2 + 0.2^2} = \sqrt{0.64 + 0.64 + 0.04} = \sqrt{1.32} = 1.1489
$$

#### Baris 5 vs Baris 4

Baris 4: `[5.0, 3.6, 1.4]`

$$
d(5,4) = \sqrt{(5.4-5.0)^2 + (3.9-3.6)^2 + (1.7-1.4)^2}
$$

$$
= \sqrt{0.4^2 + 0.3^2 + 0.3^2} = \sqrt{0.16 + 0.09 + 0.09} = \sqrt{0.34} = 0.5831
$$

Perhitungan yang sama dilakukan untuk **seluruh 149 baris lainnya**.

---

### 3.4 Menentukan 3 Tetangga Terdekat (k = 3)

Setelah seluruh jarak dihitung dan diurutkan dari terkecil, diperoleh **3 tetangga terdekat**:

| Peringkat | Baris | Jarak | Data (SL, SW, PL) | petal_width |
|---|---|---|---|---|
| 1 | Baris 10 | 0.2828 | [5.4, 3.7, 1.5] | 0.2 |
| 2 | Baris 48 | 0.3000 | [5.3, 3.7, 1.5] | 0.2 |
| 3 | Baris 18 | 0.3162 | [5.7, 3.8, 1.7] | 0.3 |

**Verifikasi manual baris 5 vs baris 10:**

Baris 10: `[5.4, 3.7, 1.5]`

$$
d(5,10) = \sqrt{(5.4-5.4)^2 + (3.9-3.7)^2 + (1.7-1.5)^2} = \sqrt{0 + 0.04 + 0.04} = \sqrt{0.08} = 0.2828
$$

---

### 3.5 Imputasi Missing Value (KNN Regression)

Karena nilai yang diisi berupa numerik, digunakan **rata-rata** dari `petal_width` ketiga tetangga:

$$
\hat{x} = \frac{0.2 + 0.2 + 0.3}{3} = \frac{0.7}{3} = 0.2333
$$

**Hasil:** Missing value `petal_width` pada baris 5 diisi dengan $\boxed{0.2333}$

Nilai asli sebelum dihilangkan: **0.4**

---

### 3.6 Kode Python — Imputasi KNN pada Iris

```python
import pandas as pd
import numpy as np

# Load data
iris = pd.read_csv("IRIS.csv")

# Simpan nilai asli sebelum dihilangkan
target_idx = 5
col_missing = 'petal_width'
nilai_asli = iris.loc[target_idx, col_missing]

# Buat missing value buatan
iris_knn = iris.copy()
iris_knn.loc[target_idx, col_missing] = np.nan

# Fitur yang dipakai untuk menghitung jarak (tanpa kolom yang missing)
fitur = ['sepal_length', 'sepal_width', 'petal_length']

# Hitung jarak ke semua baris lain
hasil_jarak = []
for i in range(len(iris_knn)):
    if i == target_idx:
        continue
    d = np.sqrt(((iris_knn.loc[target_idx, fitur] - iris_knn.loc[i, fitur]) ** 2).sum())
    hasil_jarak.append((i, d, iris.loc[i, col_missing]))

# Urutkan berdasarkan jarak terkecil
hasil_jarak = sorted(hasil_jarak, key=lambda x: x[1])

# Ambil 3 tetangga terdekat
k = 3
tetangga = hasil_jarak[:k]

# Imputasi dengan rata-rata (KNN Regression)
imputasi = np.mean([x[2] for x in tetangga])

print("Nilai asli petal_width:", nilai_asli)
print("\n3 tetangga terdekat:")
for t in tetangga:
    print(f"  Baris {t[0]}: jarak = {t[1]:.4f}, petal_width = {t[2]}")
print(f"\nHasil imputasi KNN (k=3): {imputasi:.4f}")
```

**Output yang diharapkan:**

```
Nilai asli petal_width: 0.4

3 tetangga terdekat:
  Baris 10: jarak = 0.2828, petal_width = 0.2
  Baris 48: jarak = 0.3000, petal_width = 0.2
  Baris 18: jarak = 0.3162, petal_width = 0.3

Hasil imputasi KNN (k=3): 0.2333
```

---

### 3.7 Hasil Gambar — Dataset Iris

#### Histogram Fitur Iris (Sebelum Scaling)

![Histogram Iris sebelum scaling](Assets/Pertemuan3/SebelumScalling.png)

#### Histogram Fitur Iris (Sesudah Scaling)

![Histogram Iris sesudah scaling](Assets/Pertemuan3/SesudahScalling.png)

#### Workflow Pengukuran Jarak di Orange

![Workflow pengukuran jarak Iris di Orange](Assets/Pertemuan3/DataIrisOrangePengukuranJarak.png)

---

## 4. Penerapan pada Dataset Bank (Data Campuran)

Dataset Bank merupakan **data campuran** karena memiliki kolom numerik, ordinal, dan kategorikal. Oleh karena itu, jarak **tidak bisa langsung** dihitung dengan Euclidean untuk semua kolom — setiap tipe data harus diperlakukan sesuai jenisnya.

### 4.1 Menentukan Tipe Variabel

Kolom-kolom yang digunakan dalam perhitungan jarak:

| Kolom | Tipe Data | Keterangan |
|---|---|---|
| `age` | **Numerik** | Usia nasabah |
| `education` | **Ordinal** | primary < secondary < tertiary |
| `marital` | **Kategorikal** | married, single, divorced |
| `housing` | **Kategorikal** | yes, no |
| `loan` | **Kategorikal** | yes, no |
| `balance` | **Numerik (target)** | Saldo — **kolom yang akan diimputasi** |

Karena `balance` berupa numerik, maka pengisian missing value menggunakan **KNN Regression**.

### 4.2 Membuat Missing Value Buatan

Dataset Bank tidak memiliki missing value, sehingga dibuat satu missing value buatan:

- **Baris indeks 5**
- **Kolom `balance`**

Data asli baris 5:

| age | education | marital | housing | loan | balance |
|---|---|---|---|---|---|
| 42 | tertiary | single | yes | yes | **0** |

Setelah dihilangkan:

| age | education | marital | housing | loan | balance |
|---|---|---|---|---|---|
| 42 | tertiary | single | yes | yes | **?** |

Karena `balance` kosong, maka perhitungan jarak menggunakan kolom: `age`, `education`, `marital`, `housing`, `loan`.

---

### 4.3 Langkah 1 — Konversi Ordinal ke Numerik

Kolom `education` (ordinal) dikonversi menjadi numerik:

$$
z = \frac{r - 1}{m - 1}
$$

| Level | $r$ | Hasil |
|---|---|---|
| primary | 1 | $(1-1)/(3-1) = 0$ |
| secondary | 2 | $(2-1)/(3-1) = 0.5$ |
| tertiary | 3 | $(3-1)/(3-1) = 1$ |

Baris 5 memiliki `education = tertiary`, maka nilai numeriknya = **1.0**.

---

### 4.4 Langkah 2 — Normalisasi Data Numerik (Min-Max)

Kolom `age` dinormalisasi agar skalanya setara dengan kolom ordinal:

$$
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

Dari dataset Bank: `age_min = 23`, `age_max = 60`.

Baris 5: `age = 42`

$$
age'_5 = \frac{42 - 23}{60 - 23} = \frac{19}{37} = 0.5135
$$

---

### 4.5 Langkah 3 — Perhitungan Jarak Manual

Sekarang kita hitung jarak baris 5 ke beberapa baris lain, dengan menggabungkan:

- **Jarak numerik + ordinal** (Euclidean) dari kolom `age` dan `education`
- **Jarak kategorikal** dari kolom `marital`, `housing`, `loan`

#### Baris 5 vs Baris 21

Baris 21: `age=43, education=tertiary, marital=single, housing=yes, loan=no`

**Numerik + Ordinal:**

$$
age'_{21} = \frac{43-23}{37} = 0.5405 \quad;\quad edu'_{21} = 1.0
$$

$$
d_{num+ord} = \sqrt{(0.5135 - 0.5405)^2 + (1.0 - 1.0)^2} = \sqrt{0.0007 + 0} = 0.0270
$$

**Kategorikal:**

| Kolom | Baris 5 | Baris 21 | Sama? |
|---|---|---|---|
| marital | single | single | ✓ |
| housing | yes | yes | ✓ |
| loan | yes | no | ✗ |

$P = 3$, $M = 2$:

$$
d_{kat} = \frac{3-2}{3} = 0.3333
$$

**Jarak Total:**

$$
d_{total} = 0.0270 + 0.3333 = \boxed{0.3604}
$$

---

#### Baris 5 vs Baris 96

Baris 96: `age=30, education=tertiary, marital=single, housing=yes, loan=yes`

**Numerik + Ordinal:**

$$
age'_{96} = \frac{30-23}{37} = 0.1892 \quad;\quad edu'_{96} = 1.0
$$

$$
d_{num+ord} = \sqrt{(0.5135 - 0.1892)^2 + (1.0 - 1.0)^2} = \sqrt{0.1052 + 0} = 0.3243
$$

**Kategorikal:**

| Kolom | Baris 5 | Baris 96 | Sama? |
|---|---|---|---|
| marital | single | single | ✓ |
| housing | yes | yes | ✓ |
| loan | yes | yes | ✓ |

$P = 3$, $M = 3$:

$$
d_{kat} = \frac{3-3}{3} = 0
$$

**Jarak Total:**

$$
d_{total} = 0.3243 + 0 = \boxed{0.3243}
$$

---

#### Baris 5 vs Baris 51

Baris 51: `age=39, education=tertiary, marital=divorced, housing=yes, loan=yes`

**Numerik + Ordinal:**

$$
age'_{51} = \frac{39-23}{37} = 0.4324 \quad;\quad edu'_{51} = 1.0
$$

$$
d_{num+ord} = \sqrt{(0.5135 - 0.4324)^2 + (1.0 - 1.0)^2} = \sqrt{0.0066 + 0} = 0.0811
$$

**Kategorikal:**

| Kolom | Baris 5 | Baris 51 | Sama? |
|---|---|---|---|
| marital | single | divorced | ✗ |
| housing | yes | yes | ✓ |
| loan | yes | yes | ✓ |

$P = 3$, $M = 2$:

$$
d_{kat} = \frac{3-2}{3} = 0.3333
$$

**Jarak Total:**

$$
d_{total} = 0.0811 + 0.3333 = \boxed{0.4144}
$$

---

#### Baris 5 vs Baris 76

Baris 76: `age=39, education=tertiary, marital=married, housing=yes, loan=yes`

**Numerik + Ordinal:**

$$
age'_{76} = \frac{39-23}{37} = 0.4324 \quad;\quad edu'_{76} = 1.0
$$

$$
d_{num+ord} = \sqrt{(0.5135 - 0.4324)^2 + (1.0 - 1.0)^2} = \sqrt{0.0066 + 0} = 0.0811
$$

**Kategorikal:**

| Kolom | Baris 5 | Baris 76 | Sama? |
|---|---|---|---|
| marital | single | married | ✗ |
| housing | yes | yes | ✓ |
| loan | yes | yes | ✓ |

$P = 3$, $M = 2$:

$$
d_{kat} = \frac{3-2}{3} = 0.3333
$$

**Jarak Total:**

$$
d_{total} = 0.0811 + 0.3333 = \boxed{0.4144}
$$

---

#### Baris 5 vs Baris 13

Baris 13: `age=46, education=tertiary, marital=single, housing=yes, loan=no`

**Numerik + Ordinal:**

$$
age'_{13} = \frac{46-23}{37} = 0.6216 \quad;\quad edu'_{13} = 1.0
$$

$$
d_{num+ord} = \sqrt{(0.5135 - 0.6216)^2 + (1.0 - 1.0)^2} = \sqrt{0.0117 + 0} = 0.1081
$$

**Kategorikal:**

| Kolom | Baris 5 | Baris 13 | Sama? |
|---|---|---|---|
| marital | single | single | ✓ |
| housing | yes | yes | ✓ |
| loan | yes | no | ✗ |

$P = 3$, $M = 2$:

$$
d_{kat} = \frac{3-2}{3} = 0.3333
$$

**Jarak Total:**

$$
d_{total} = 0.1081 + 0.3333 = \boxed{0.4414}
$$

---

Perhitungan yang sama dilakukan untuk **seluruh baris lainnya**.

---

### 4.6 Menentukan 3 Tetangga Terdekat (k = 3)

Setelah seluruh jarak dihitung dan diurutkan:

| Peringkat | Baris | $d_{num+ord}$ | $d_{kat}$ | $d_{total}$ | balance |
|---|---|---|---|---|---|
| 1 | Baris 96 | 0.3243 | 0 | **0.3243** | 880 |
| 2 | Baris 21 | 0.0270 | 0.3333 | **0.3604** | 2067 |
| 3 | Baris 51 | 0.0811 | 0.3333 | **0.4144** | 517 |

**Perhatikan:**

- Baris 96 memiliki jarak numerik+ordinal besar (0.3243) tapi kategorikal semua sama (d_kat = 0), sehingga total jarak kecil
- Baris 21 memiliki jarak numerik+ordinal kecil (0.0270) tapi ada 1 kolom kategorikal berbeda (d_kat = 0.3333)

Ini menunjukkan pentingnya **memperhitungkan kedua tipe jarak** pada data campuran.

---

### 4.7 Imputasi Missing Value (KNN Regression)

$$
\hat{balance} = \frac{880 + 2067 + 517}{3} = \frac{3464}{3} = 1154.67
$$

**Hasil:** Missing value `balance` pada baris 5 diisi dengan $\boxed{1154.67}$

Nilai asli sebelum dihilangkan: **0**

---

### 4.8 Kode Python — Imputasi KNN pada Data Campuran Bank

```python
import pandas as pd
import numpy as np

# Load data
bank = pd.read_csv("bank.csv")

# Simpan nilai asli sebelum dihilangkan
target_idx = 5
col_missing = 'balance'
nilai_asli = bank.loc[target_idx, col_missing]

# Buat missing value buatan
bank_knn = bank.copy()
bank_knn.loc[target_idx, col_missing] = np.nan

# === KONVERSI ORDINAL ===
edu_order = {'primary': 1, 'secondary': 2, 'tertiary': 3}
m_edu = 3

def ord_norm(val):
    return (edu_order[val] - 1) / (m_edu - 1)

# Filter baris yang education-nya valid
bank_knn = bank_knn[bank_knn['education'].isin(edu_order.keys())].copy()
bank_knn = bank_knn.reset_index(drop=True)

# Cari ulang indeks target setelah reset
missing_idx = bank_knn[bank_knn[col_missing].isna()].index[0]

# === NORMALISASI NUMERIK (Min-Max) ===
age_min = bank_knn['age'].min()
age_max = bank_knn['age'].max()

def age_norm(val):
    return (val - age_min) / (age_max - age_min)

# === PERHITUNGAN JARAK ===
cat_cols = ['marital', 'housing', 'loan']
hasil_jarak = []

for i in range(len(bank_knn)):
    if i == missing_idx:
        continue

    # Jarak numerik + ordinal (Euclidean)
    d_numord = np.sqrt(
        (age_norm(bank_knn.loc[missing_idx, 'age']) - age_norm(bank_knn.loc[i, 'age']))**2 +
        (ord_norm(bank_knn.loc[missing_idx, 'education']) - ord_norm(bank_knn.loc[i, 'education']))**2
    )

    # Jarak kategorikal
    P = len(cat_cols)
    M = sum(bank_knn.loc[missing_idx, c] == bank_knn.loc[i, c] for c in cat_cols)
    d_cat = (P - M) / P

    # Jarak total
    d_total = d_numord + d_cat

    hasil_jarak.append((i, d_total, bank_knn.loc[i, col_missing] if not pd.isna(bank_knn.loc[i, col_missing]) else 0))

# Urutkan berdasarkan jarak terkecil
hasil_jarak = sorted(hasil_jarak, key=lambda x: x[1])

# Ambil 3 tetangga terdekat
k = 3
tetangga = hasil_jarak[:k]

# Imputasi dengan rata-rata (KNN Regression)
imputasi = np.mean([x[2] for x in tetangga])

print("Nilai asli balance:", nilai_asli)
print("\n3 tetangga terdekat:")
for t in tetangga:
    print(f"  Baris {t[0]}: d_total = {t[1]:.4f}, balance = {t[2]}")
print(f"\nHasil imputasi KNN (k=3): {imputasi:.2f}")
```

---

### 4.9 Hasil Gambar — Dataset Bank

#### Data Iris di Orange dengan Pengukuran Jarak

![Workflow pengukuran jarak di Orange](Assets/Pertemuan3/DataIrisOrangePengukuranJarak.png)

#### Bukti Data CSV ke PostgreSQL

![CSV ke PostgreSQL](Assets/Pertemuan3/Gambar%20Csv%20ke%20PostgreeSQL.png)

---

## 5. Ringkasan Langkah KNN Imputation

### 5.1 Untuk Data Numerik (Iris)

```
1. Cek missing value → tidak ada → buat missing value buatan
2. Tentukan baris target dan kolom yang kosong
3. Hitung jarak Euclidean ke semua baris lain (tanpa kolom yang kosong)
4. Urutkan jarak dari terkecil
5. Ambil k=3 tetangga terdekat
6. Isi missing value = rata-rata nilai kolom target dari 3 tetangga
```

### 5.2 Untuk Data Campuran (Bank)

```
1. Cek missing value → tidak ada → buat missing value buatan
2. Tentukan tipe setiap kolom: numerik, ordinal, atau kategorikal
3. Konversi ordinal ke numerik dengan rumus z = (r-1)/(m-1)
4. Normalisasi kolom numerik dengan Min-Max Scaling
5. Hitung jarak numerik+ordinal menggunakan Euclidean
6. Hitung jarak kategorikal menggunakan (P-M)/P
7. Jumlahkan: d_total = d_num+ord + d_kat
8. Urutkan jarak dari terkecil
9. Ambil k=3 tetangga terdekat
10. Isi missing value = rata-rata nilai target dari 3 tetangga (KNN Regression)
```

---

## 6. Kesimpulan

1. **Dataset Iris** termasuk data numerik, sehingga missing value dapat diimputasi langsung menggunakan **Euclidean distance** → hasilnya: `petal_width = 0.2333`

2. **Dataset Bank** merupakan data campuran, sehingga:
   - Data ordinal (`education`) harus diubah menjadi numerik terlebih dahulu menggunakan rumus $z = (r-1)/(m-1)$
   - Data kategorikal (`marital`, `housing`, `loan`) dihitung menggunakan rasio ketidaksamaan $(P-M)/P$
   - Jarak total = jarak numerik/ordinal + jarak kategorikal
   - Hasilnya: `balance = 1154.67`

3. Jika ada missing value pada satu kolom, maka **kolom tersebut tidak diikutkan** dalam perhitungan jarak

4. Setelah semua jarak dihitung, dipilih **k = 3 tetangga terdekat**

5. Karena nilai yang diisi berupa numerik, maka digunakan **KNN Regression** (rata-rata dari tetangga terdekat)

