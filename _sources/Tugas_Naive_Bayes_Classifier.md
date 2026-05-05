# Tugas | Analisa Data Menggunakan Naive Bayes (A)
## NIM 240411100144
## Nama: Zakaria Mujur Prasetyo
## Mata Kuliah Penambangan Data A
## Dosen Pengampu: Mula'ab, S.Si., M.Kom

## Daftar Isi
```{dropdown} Klik untuk membuka Daftar Isi
:open:

1. [Pendahuluan](#pendahuluan)
2. [Teori Naive Bayes](#teori-naive-bayes)
3. [Informasi Dataset](#informasi-dataset)
4. [Eksplorasi Data](#eksplorasi-data)
5. [Implementasi Model KNIME dengan Python Script](#implementasi-model-knime-dengan-python-script)
6. [Kesimpulan](#kesimpulan)
7. [Referensi](#referensi)
```

## Pendahuluan

Pada tugas kali ini, saya melakukan analisis data menggunakan algoritma Naive Bayes Classifier. Untuk pengerjaannya, saya menggunakan pendekatan yang sedikit berbeda. Saya menggabungkan keunggulan KNIME Analytics Platform dengan fleksibilitas bahasa pemrograman Python melalui penggunaan node Python Script.

Alasan saya memilih Naive Bayes adalah karena algoritma ini cukup sederhana namun kinerjanya sangat efektif. Algoritma ini berakar dari Teorema Bayes dan bekerja dengan asumsi bahwa setiap fitur berdiri sendiri dan tidak saling mempengaruhi probabilitas kelas target.

```{admonition} Tujuan Tugas
:class: note

1. Melakukan eksplorasi data secara visual
2. Membangun model klasifikasi Naive Bayes menggunakan kombinasi KNIME dan Python
3. Melatih model Gaussian Naive Bayes untuk mendeteksi kelas pada dataset Iris
4. Mengukur kinerja model melalui metrik Accuracy, Precision, Recall, dan F1-Score
5. Melihat langsung hasil prediksi pada data testing
```

## Teori Naive Bayes

### Bayesian Theorem

Cara kerja Naive Bayes Classifier bertumpu pada Teorema Bayes. Teorema ini menghitung seberapa besar probabilitas posterior dari sebuah kelas jika diberikan suatu data pengamatan. Jika saya memiliki training data X, maka posteriori probabilitas dari kelas H (P(H|X)) bisa dicari dengan rumus Teorema Bayes:

$$
P(C \mid \mathbf{X}) = \frac{P(\mathbf{X} \mid C) \cdot P(C)}{P(\mathbf{X})}
$$

Keterangan:
* $P(C \mid \mathbf{X})$ adalah Posterior atau probabilitas kelas C jika diketahui data X.
* $P(\mathbf{X} \mid C)$ adalah Likelihood atau probabilitas kemunculan data X pada kelas C.
* $P(C)$ adalah Prior atau probabilitas awal sebuah kelas C.
* $P(\mathbf{X})$ adalah Evidence atau probabilitas data X secara keseluruhan.

### Asumsi Naive (Independensi)

Algoritma ini mendapatkan nama "Naive" karena memiliki asumsi yang sangat kuat bahwa seluruh fitur saling bebas. Artinya, nilai dari sebuah fitur tidak ada kaitannya dengan fitur lainnya:

$$
P(\mathbf{X} \mid C_j) = \prod_{k=1}^{n} P(x_k \mid C_j)
$$

Dengan begitu, tugas saya dalam proses klasifikasi adalah mencari nilai maksimal dari:

$$
P(C_j \mid \mathbf{X}) = P(\mathbf{X} \mid C_j) \cdot P(C_j)
$$

### Gaussian Naive Bayes

Karena data yang saya gunakan bernilai kontinu (angka desimal), perhitungan probabilitasnya menggunakan distribusi Gaussian atau distribusi normal. Perhitungannya menggunakan mean dan standar deviasi:

$$
P(x_k \mid C_j) = g(x_k, \mu_{C_j}, \sigma_{C_j}) = \frac{1}{\sqrt{2\pi} \cdot \sigma} \cdot e^{-\frac{(x - \mu)^2}{2\sigma^2}}
$$

Di dalam library sklearn, metode GaussianNB secara otomatis akan menghitung mean dan variance setiap fitur per kelas berdasarkan data training yang saya berikan.

## Informasi Dataset

Untuk tugas ini, saya memilih Dataset Iris. Dataset ini sangat populer dalam dunia machine learning dan diperkenalkan pertama kali oleh Ronald A. Fisher. Di dalamnya terdapat ukuran morfologi dari tiga spesies bunga iris.

### Deskripsi Umum

* Jumlah Sampel: 150 baris
* Jumlah Fitur: 4 fitur berupa numerik kontinu
* Jumlah Kelas: 3 kelas
* Target Kelas: Setosa, Versicolor, dan Virginica
* Missing Values: Tidak ada

Dari 150 sampel tersebut, distribusinya sangat seimbang. Masing-masing kelas (Setosa, Versicolor, Virginica) memiliki persis 50 sampel.

### Penjelasan Fitur

1. Sepal Length: Panjang kelopak luar bunga dalam satuan cm
2. Sepal Width: Lebar kelopak luar bunga dalam satuan cm
3. Petal Length: Panjang kelopak dalam bunga dalam satuan cm
4. Petal Width: Lebar kelopak dalam bunga dalam satuan cm

Saya menggunakan dataset ini karena datanya sudah seimbang sehingga saya tidak perlu melakukan teknik penyeimbangan data. Selain itu, semua fiturnya bertipe numerik kontinu yang sangat pas untuk diterapkan pada algoritma Gaussian Naive Bayes.

## Eksplorasi Data

Sebelum masuk ke tahap modeling, saya melakukan eksplorasi data menggunakan Python untuk memahami karakteristik fitur yang ada.

### Distribusi Fitur per Kelas

Saya membuat plot histogram untuk melihat bagaimana sebaran data pada masing-masing fitur.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribusi Fitur Dataset Iris per Kelas', fontsize=16, fontweight='bold')
colors = {'Setosa': '#2ecc71', 'Versicolor': '#3498db', 'Virginica': '#e74c3c'}

for idx, col in enumerate(iris.feature_names):
    ax = axes[idx // 2, idx % 2]
    for species_name, color in colors.items():
        subset = df[df['species_name'] == species_name]
        ax.hist(subset[col], bins=15, alpha=0.6, label=species_name, color=color, edgecolor='white')
    ax.set_title(col.title(), fontsize=12, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Frekuensi')
    ax.legend()
plt.tight_layout()
plt.show()
```

Dari hasil plot yang saya buat, terlihat jelas bahwa spesies Setosa memiliki ukuran kelopak dalam (petal) yang lebih kecil dibandingkan dua spesies lainnya. Sedangkan untuk Versicolor dan Virginica, ukurannya sedikit tumpang tindih.

### Scatter Plot

Saya juga menggunakan scatter plot untuk melihat hubungan antar fitur.

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for species_name, color in colors.items():
    subset = df[df['species_name'] == species_name]
    axes[0].scatter(subset['sepal length (cm)'], subset['sepal width (cm)'],
                    c=color, label=species_name, alpha=0.7, edgecolors='white', s=60)
    axes[1].scatter(subset['petal length (cm)'], subset['petal width (cm)'],
                    c=color, label=species_name, alpha=0.7, edgecolors='white', s=60)
axes[0].set_title('Sepal Length vs Sepal Width')
axes[1].set_title('Petal Length vs Petal Width')
for ax in axes:
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

Berdasarkan scatter plot, fitur petal (length dan width) lebih ampuh untuk membedakan ketiga kelas tersebut dibandingkan fitur sepal.

## Implementasi Model KNIME dengan Python Script

Untuk tahap pemodelannya, saya merancang sebuah workflow di KNIME yang memanfaatkan node Python Script. Pendekatan ini sangat efektif karena saya bisa mengkombinasikan kekuatan antarmuka visual KNIME dengan pustaka scikit-learn dari Python.

### A. Strategi Workflow Final

Dalam merancang workflow ini, saya menggunakan **4 buah node Python Script**. Saya memilih cara ini karena paling aman dan minim error. Jika saya hanya menggunakan satu Python Script lalu menarik banyak garis output, seringkali muncul error berupa:

```text
Invalid port index 1, only 1 output_table is available
```

Error tersebut terjadi karena KNIME mendeteksi bahwa saya memanggil `knio.output_tables[1]` hingga ke-3, padahal dari segi tampilan node tetap dihitung sebagai satu output port. Oleh karena itu, memisahkan script ke dalam 4 node berbeda adalah solusi terbaik.

### B. Bentuk Workflow di KNIME

Alur kerja (workflow) yang saya bangun berbentuk seperti ini:

```text
                 ┌── Python Script Ringkasan Data   ──> Table View Ringkasan
                 │
CSV Reader ──────┼── Python Script Evaluasi Model   ──> Table View Evaluasi
                 │
                 ├── Python Script Confusion Matrix ──> Table View Confusion
                 │
                 └── Python Script Prediksi Testing ──> Table View Prediksi
```

Berikut adalah tangkapan layar keseluruhan workflow yang saya buat:

![Tampilan Workflow Lengkap di KNIME](Assets/Tugas/Tugas_NaiveBayesClassifer/workflow.png)

### C. Node yang Digunakan

Dalam workflow ini, saya menggunakan kombinasi node pembaca data, pengeksekusi script, dan penampil tabel. Berikut daftar lengkapnya:

1. **CSV Reader**: Node awal untuk membaca file dataset `IRIS.csv`.
2. **Python Script Ringkasan Data**: Node untuk memproses dan menampilkan total data beserta pembagiannya (training dan testing).
3. **Python Script Evaluasi Model**: Node khusus untuk menghitung accuracy, precision, recall, dan f1-score.
4. **Python Script Confusion Matrix**: Node untuk menyusun matriks kebingungan (confusion matrix) dari hasil prediksi.
5. **Python Script Prediksi Testing**: Node untuk menampilkan rincian hasil tebakan model pada setiap baris data testing.
6. **Table View (4 buah)**: Node untuk memunculkan output dari tiap-tiap Python Script secara visual.

### D. Langkah Membuat Workflow

Pertama, saya mengonfigurasi node **CSV Reader** untuk mengambil file `IRIS.csv`. Saya memastikan bahwa kolom-kolom seperti `sepal_length`, `sepal_width`, `petal_length`, `petal_width`, dan `species` terbaca dengan baik. Setelah itu, saya tekan Execute agar data siap disalurkan.

Dari node CSV Reader tersebut, saya menarik garis ke empat node **Python Script** yang telah saya siapkan. Setiap Python script hanya mengandalkan satu output table saja untuk mencegah error indeks. Selanjutnya, saya hubungkan masing-masing script tersebut ke node **Table View** untuk melihat hasilnya.

### E. Kode Python Script 1: Ringkasan Data

Pada node script pertama ini, tujuan saya adalah memberikan informasi dasar mengenai dataset dan skema pembagian data yang saya gunakan.

```python
import knime.scripting.io as knio
import pandas as pd
from sklearn.model_selection import train_test_split

# Saya membaca data yang dikirimkan oleh CSV Reader
df = knio.input_tables[0].to_pandas()

feature_cols = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

target_col = "species"

X = df[feature_cols]
y = df[target_col]

# Membagi data menjadi 80% training dan 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

ringkasan_df = pd.DataFrame({
    "keterangan": [
        "Jumlah seluruh data",
        "Jumlah data training",
        "Jumlah data testing",
        "Persentase training",
        "Persentase testing",
        "Kolom target",
        "Model yang digunakan"
    ],
    "nilai": [
        len(df),
        len(X_train),
        len(X_test),
        "80%",
        "20%",
        target_col,
        "Gaussian Naive Bayes"
    ]
})

ringkasan_df = ringkasan_df.astype(str)

knio.output_tables[0] = knio.Table.from_pandas(ringkasan_df)
```

Berikut adalah hasil tabel ringkasan data yang muncul di KNIME:

![Output Table View Ringkasan Data](Assets/Tugas/Tugas_NaiveBayesClassifer/TableViewRingkasanData.png)

### F. Kode Python Script 2: Evaluasi Model

Pada script kedua, saya melatih model Gaussian Naive Bayes lalu langsung menghitung kinerja model tersebut menggunakan metrik standar.

```python
import knime.scripting.io as knio
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Proses baca data
df = knio.input_tables[0].to_pandas()

feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
target_col = "species"

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Tahap pelatihan dan prediksi
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Perhitungan metrik evaluasi
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

evaluasi_df = pd.DataFrame({
    "metrik": ["Accuracy", "Precision Weighted", "Recall Weighted", "F1 Score Weighted"],
    "nilai": [round(accuracy, 4), round(precision, 4), round(recall, 4), round(f1, 4)]
})

knio.output_tables[0] = knio.Table.from_pandas(evaluasi_df)
```

Hasil metrik evaluasinya terlihat sangat baik, sebagaimana ditunjukkan pada gambar berikut:

![Output Table View Evaluasi Model](Assets/Tugas/Tugas_NaiveBayesClassifer/TableViewEvaluasiModel.png)

### G. Kode Python Script 3: Confusion Matrix

Untuk melihat sebaran prediksi yang benar dan yang meleset, saya menyusun confusion matrix.

```python
import knime.scripting.io as knio
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

df = knio.input_tables[0].to_pandas()

feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
target_col = "species"

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

confusion_df = pd.DataFrame(
    cm,
    index=[f"Actual {cls}" for cls in model.classes_],
    columns=[f"Predicted {cls}" for cls in model.classes_]
)

confusion_df = confusion_df.reset_index()
confusion_df = confusion_df.rename(columns={"index": "Actual / Predicted"})

knio.output_tables[0] = knio.Table.from_pandas(confusion_df)
```

Berikut tampilan matriks kebingungannya:

![Output Table View Confusion Matrix](Assets/Tugas/Tugas_NaiveBayesClassifer/TableViewConfusionMatrix.png)

#### Solusi Jika Confusion Matrix Hanya Muncul RowID

Terkadang, Table View hanya menampilkan RowID tanpa memperlihatkan kolom datanya. Padahal datanya sudah ada di sana. Untuk mengatasinya, saya biasa membuka konfigurasi Table View lalu mengubah pengaturan kolom. Pada tab Wildcard, saya menampilkan semua kolom dengan mengetik pola bintang lalu menekan apply. Atau bisa juga melalui tab Manual dengan mencentang semua kolom yang ingin ditampilkan.

### H. Kode Python Script 4: Prediksi Testing

Terakhir, saya ingin melihat secara rinci data mana saja yang tebakannya benar dan mana yang salah, beserta probabilitas keyakinan model.

```python
import knime.scripting.io as knio
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = knio.input_tables[0].to_pandas()

feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
target_col = "species"

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

prediksi_df = X_test.copy().reset_index(drop=True)
prediksi_df["actual_species"] = y_test.reset_index(drop=True)
prediksi_df["predicted_species"] = y_pred

# Saya tandai mana yang tebakannya benar dan salah
prediksi_df["status_prediksi"] = np.where(
    prediksi_df["actual_species"] == prediksi_df["predicted_species"],
    "Benar",
    "Salah"
)

# Saya sertakan probabilitas untuk masing-masing kelas
for index, class_name in enumerate(model.classes_):
    prediksi_df[f"probabilitas_{class_name}"] = y_proba[:, index]

knio.output_tables[0] = knio.Table.from_pandas(prediksi_df)
```

Hasil tebakan model secara rinci dapat dilihat di sini:

![Output Table View Hasil Prediksi Testing](Assets/Tugas/Tugas_NaiveBayesClassifer/TableViewPrediksiTraining.png)

### Urutan Eksekusi

Agar tidak ada error, saya mengeksekusi node secara berurutan. Saya mulai dari mengeksekusi CSV Reader, kemudian menjalankan keempat node Python Script secara bersamaan atau berurutan. Setelah status node menjadi hijau, barulah saya membuka Table View satu per satu.

## Kesimpulan

Dari proses panjang yang telah saya jabarkan, saya berhasil menerapkan algoritma Gaussian Naive Bayes dengan perpaduan apik antara KNIME dan Python. Saya telah membagi data dengan perbandingan 80% untuk pelatihan dan 20% untuk pengujian.

Hasil akhirnya sangat memuaskan. Model mampu menebak kelas spesies bunga dengan akurasi yang luar biasa tinggi. Kesalahan prediksi terbilang sangat kecil, yang mana wajar karena ada kemiripan fisik antara spesies Versicolor dan Virginica. Secara keseluruhan, pemanfaatan Python Script di dalam KNIME memberikan fleksibilitas tinggi tanpa mengorbankan kerapian visual dari alur kerja.

## Referensi

1. Fisher, R.A., 1936. The Use of Multiple Measurements in Taxonomic Problems. Annals of Eugenics, 7(2): 179-188.
2. scikit-learn Naive Bayes API Documentation
3. scikit-learn Naive Bayes User Guide
4. Iris Dataset pada sklearn.datasets
5. Han, J., Kamber, M., Pei, J., 2011. Data Mining: Concepts and Techniques. Morgan Kaufmann.
6. Mulaab, Data Mining Website.
