# Tugas Analisa Data Menggunakan Random Forest

Pada tugas ini, saya menganalisa dataset **Adult** (US Census) menggunakan pendekatan **Decision Tree** dan **Random Forest** di dalam KNIME Analytics Platform. Tujuan utama dari analisa ini adalah memprediksi fitur **sex** (jenis kelamin) berdasarkan informasi sensus seperti usia, pendidikan, pekerjaan, dan lain-lain. Dokumentasi ini menjelaskan setiap tahapan workflow dari awal hingga evaluasi model.

## Gambaran Umum Workflow

![Workflow Overview](Assets/Tugas/AnalisaDataMenggunakanRandomForest/workflow_overview.png)

Workflow KNIME yang saya rancang terdiri dari beberapa blok utama yang ditandai dengan kotak kuning:

1. **Blok Input & Preprocessing** — Membaca dataset `adult.csv` dan melakukan partisi data.
2. **Blok Decision Tree** — Melatih satu pohon keputusan tunggal untuk memprediksi fitur `sex`, dilengkapi dengan visualisasi pohon dan evaluasi akurasi melalui *Scorer*.
3. **Blok Random Forest** — Melatih ensemble 50 pohon keputusan (Random Forest) untuk memprediksi fitur yang sama, termasuk evaluasi statistik ensemble dan *Scorer*.
4. **Blok Opsional Export PMML** — Menyediakan opsi untuk mengekspor model ensemble ke format PMML agar model bisa digunakan di luar KNIME.

Alur data dimulai dari **CSV Reader** → **Table Partitioner**, kemudian bercabang dua: satu jalur menuju Decision Tree Learner/Predictor, dan jalur satunya menuju Random Forest Learner/Predictor.

## Membaca Dataset dengan CSV Reader

![CSV Reader Configuration](Assets/Tugas/AnalisaDataMenggunakanRandomForest/csv_reader_config.png)

Langkah pertama adalah mengimpor dataset. Saya menggunakan node **CSV Reader** dengan konfigurasi sebagai berikut:

| Parameter | Nilai |
|:---|:---|
| **Mode** | File |
| **Source** | `knime://knime.workflow/data/adult.csv` |
| **Skip first lines of file** | 0 |
| **Comment line character** | `#` |

Dataset **Adult** (juga dikenal sebagai *Census Income Dataset*) berisi data sensus penduduk Amerika Serikat. Dataset ini mencakup berbagai atribut demografis dan ekonomi seperti umur, jenis pekerjaan (*workclass*), tingkat pendidikan, status pernikahan, pekerjaan (*occupation*), ras, jenis kelamin (*sex*), jam kerja per minggu, dan negara asal. Dalam analisa ini, kolom **sex** dipilih sebagai variabel target (output class) yang akan diprediksi oleh model.

## Pembagian Data dengan Table Partitioner

![Table Partitioner Configuration](Assets/Tugas/AnalisaDataMenggunakanRandomForest/table_partitioner_config.png)

Setelah data berhasil dibaca, saya membagi dataset menjadi dua bagian menggunakan node **Table Partitioner**. Konfigurasi yang saya gunakan:

| Parameter | Nilai |
|:---|:---|
| **First partition type** | Relative (%) |
| **Relative size** | 80 |
| **Sampling strategy** | Stratified |
| **Group column** | sex |
| **Fixed random seed** | Tidak dicentang |
| **If input table is empty** | Fail |

Saya memilih rasio **80:20** — artinya 80% data digunakan untuk melatih model (*training set*) dan 20% sisanya untuk menguji model (*test set*). 

Yang penting diperhatikan adalah saya menggunakan strategi **Stratified Sampling** dengan kolom grup **sex**. Strategi ini memastikan bahwa proporsi kelas target (`Male` dan `Female`) tetap terjaga di kedua partisi. Tanpa stratifikasi, bisa saja terjadi ketimpangan distribusi kelas yang membuat model belajar secara tidak seimbang dan menghasilkan prediksi yang bias.

## Pelatihan Decision Tree (Satu Pohon Keputusan)

![Decision Tree Learner Configuration](Assets/Tugas/AnalisaDataMenggunakanRandomForest/decision_tree_learner_config.png)

Pada blok pertama, saya melatih satu buah Decision Tree. Node **Decision Tree Learner** menerima data latih dari output pertama Table Partitioner. Konfigurasi yang saya gunakan:

| Parameter | Nilai |
|:---|:---|
| **Class column** | sex |
| **Quality measure** | Gini index |
| **Pruning method** | No pruning |
| **Reduced error pruning** | Tidak dicentang |
| **Minimum number of records per node** | 2 |
| **Number of records to store for view** | 10000 |

Saya memilih **Gini Index** sebagai metrik kualitas pemisahan. Gini Index mengukur tingkat ketidakmurnian (*impurity*) dari suatu node:

$$Gini(S) = 1 - \sum_{i=1}^{c} p_i^2$$

Di mana $p_i$ adalah probabilitas kelas $i$ pada himpunan data $S$. Semakin rendah nilai Gini, semakin murni node tersebut. Pada setiap pemisahan, algoritma memilih atribut yang menghasilkan penurunan Gini terbesar.

Saya juga memilih **No pruning** agar pohon keputusan tumbuh sepenuhnya tanpa dipangkas, sehingga bisa dilihat seluruh pola yang dipelajari model dari data.

### Visualisasi Pohon Keputusan

![Decision Tree View](Assets/Tugas/AnalisaDataMenggunakanRandomForest/decision_tree_view.png)

Setelah model selesai dilatih, saya menghubungkan output modelnya ke node **Decision Tree View (JavaScript)** untuk melihat struktur pohon keputusan yang terbentuk. Dari visualisasi di atas, dapat dilihat:

- **Root Node**: Atribut **relationship** terpilih sebagai pemisah pertama karena memiliki nilai Gini Gain tertinggi. Node akar menunjukkan prediksi default `Male (17431/26048)` — artinya dari total 26.048 data latih, 17.431 berjenis kelamin Male.
- **Percabangan berdasarkan `relationship`**:
  - `= Not-in-family` → **Male** (3545/6668)
  - `= Husband` → **Male** (10553/10553) — 100% Male, node murni
  - `= Wife` → **Female** (1239/1241) — hampir seluruhnya Female
  - `= Own-child` → **Male** (2232/4038)
  - `= Unmarried` → **Female** (2118/2761)
  - `= Other-relative` → **Male** (456/787)

Atribut `relationship` sangat informatif karena nilai-nilai seperti "Husband" dan "Wife" memiliki korelasi kuat dengan jenis kelamin. Pada cabang yang belum murni (misalnya `Not-in-family` dan `Own-child`), pohon melanjutkan pemisahan menggunakan atribut **occupation** dan **native-country** untuk memperbaiki prediksi.

Setelah visualisasi, output model juga dihubungkan ke node **Decision Tree Predictor** untuk menerapkan model ke data uji, lalu hasilnya dievaluasi menggunakan node **Scorer (JavaScript)** yang menghitung *Confusion Matrix* dan metrik akurasi.

## Pelatihan Random Forest (Ensemble 50 Pohon)

![Random Forest Learner Configuration](Assets/Tugas/AnalisaDataMenggunakanRandomForest/random_forest_learner_config.png)

Pada blok kedua, saya menggunakan pendekatan **Random Forest** — sebuah metode ensemble yang membangun banyak pohon keputusan dan menggabungkan hasilnya. Konfigurasi node **Random Forest Learner**:

| Parameter | Nilai |
|:---|:---|
| **Target column** | sex |
| **Training attributes** | Use column |
| **Attribute selection** | Manual — semua kolom di-*include* (age, workclass, dll.) |
| **Jumlah pohon (dec trees)** | 50 |
| **Minimum node size** | 2 |

Saya menggunakan seleksi atribut **Manual** dan memasukkan semua kolom ke dalam daftar *Includes* (age, workclass, dan seterusnya) sebagai fitur prediktor. Kolom **sex** otomatis menjadi target dan tidak dimasukkan sebagai fitur input.

### Mengapa Random Forest Lebih Baik dari Decision Tree Tunggal?

Random Forest mengatasi kelemahan utama Decision Tree tunggal, yaitu **overfitting**. Berikut perbandingannya:

| Aspek | Decision Tree | Random Forest |
|:---|:---|:---|
| **Jumlah model** | 1 pohon | 50 pohon (ensemble) |
| **Risiko overfitting** | Tinggi | Rendah (rata-rata dari banyak pohon) |
| **Stabilitas prediksi** | Sensitif terhadap perubahan data | Lebih stabil dan robust |
| **Mekanisme** | Satu pohon membuat keputusan | Voting mayoritas dari 50 pohon |

Setiap pohon dalam Random Forest dilatih menggunakan:
- **Bootstrap sampling**: Setiap pohon dilatih pada sampel acak (*with replacement*) dari data latih.
- **Random feature selection**: Pada setiap pemisahan node, hanya subset acak dari fitur yang dipertimbangkan.

Prediksi akhir ditentukan melalui **majority voting** — kelas yang paling banyak dipilih oleh 50 pohon menjadi prediksi final.

### Statistik Tree Ensemble

![Tree Ensemble Statistics](Assets/Tugas/AnalisaDataMenggunakanRandomForest/tree_ensemble_statistics.png)

Node **Tree Ensemble Statistics** menampilkan ringkasan statistik dari 50 pohon yang dibangun oleh Random Forest. Dari tabel statistik di atas, dapat diringkas:

| Metrik | Nilai |
|:---|:---|
| **Number of models** | 50 |
| **Minimal depth** | 10 |
| **Maximal depth** | 10 |
| **Average depth** | 10 |
| **Minimal number of nodes** | 373 |
| **Maximal number of nodes** | 787 |
| **Average number of nodes** | 582.04 |

Semua 50 pohon memiliki kedalaman yang seragam yaitu **10 level**, yang menunjukkan bahwa pohon-pohon tersebut tumbuh cukup dalam untuk menangkap pola-pola kompleks dalam data. Jumlah node bervariasi dari **373 hingga 787** node per pohon (rata-rata 582 node), yang mencerminkan efek *random feature selection* — setiap pohon memiliki struktur yang berbeda karena hanya mempertimbangkan subset acak dari fitur pada setiap pemisahan.

### Evaluasi Random Forest dengan Scorer

![Random Forest Scorer - Confusion Matrix](Assets/Tugas/AnalisaDataMenggunakanRandomForest/random_forest_scorer.png)

Node **Scorer (JavaScript)** mengevaluasi prediksi Random Forest pada data uji. Dari *Confusion Matrix* yang dihasilkan:

| | Prediksi: Female | Prediksi: Male |
|:---|:---:|:---:|
| **Aktual: Female** | **1747** (True Positive) | 407 (False Negative) |
| **Aktual: Male** | 587 (False Positive) | **3772** (True Negative) |

Dari matriks ini, dapat dihitung metrik evaluasi:

- **Total data uji**: 1747 + 407 + 587 + 3772 = **6.513**
- **Prediksi benar**: 1747 + 3772 = **5.519**
- **Akurasi keseluruhan**: 5519 / 6513 = **84,7%**
- **Precision (Female)**: 1747 / (1747 + 587) = **74,8%**
- **Recall (Female)**: 1747 / (1747 + 407) = **81,1%**
- **Precision (Male)**: 3772 / (3772 + 407) = **90,3%**
- **Recall (Male)**: 3772 / (3772 + 587) = **86,5%**

Model Random Forest berhasil memprediksi jenis kelamin dengan akurasi **84,7%**. Performa prediksi untuk kelas **Male** lebih tinggi (precision 90,3%) dibanding kelas **Female** (precision 74,8%), yang kemungkinan disebabkan oleh distribusi data yang tidak seimbang — jumlah Male lebih banyak dari Female dalam dataset Adult.

## Opsional: Export Model ke PMML

Blok terakhir dalam workflow menyediakan opsi untuk mengekspor model Random Forest ke format **PMML** (*Predictive Model Markup Language*):

1. **Tree Ensemble Model Extract** — Mengekstrak model ensemble dari Random Forest Learner.
2. **Table to PMML Ensemble** — Mengonversi model tersebut ke format PMML standar.

Format PMML memungkinkan model yang sudah dilatih untuk di-*deploy* dan digunakan di platform lain di luar KNIME, menjadikan model lebih portabel dan siap produksi.

## Kesimpulan

Melalui analisa ini, saya membandingkan dua pendekatan klasifikasi pada dataset Adult:

1. **Decision Tree tunggal** memberikan model yang mudah diinterpretasi — dari visualisasi terlihat bahwa atribut `relationship` menjadi pemisah utama karena korelasinya yang kuat dengan jenis kelamin. Namun, pohon tunggal tanpa pruning rentan terhadap overfitting pada dataset besar.

2. **Random Forest dengan 50 pohon** menghasilkan akurasi **84,7%** pada data uji, dengan performa prediksi yang lebih baik untuk kelas Male (precision 90,3%) dibanding Female (precision 74,8%). Kedalaman seragam (10 level) dan variasi jumlah node (373–787) menunjukkan bahwa setiap pohon belajar pola yang berbeda berkat mekanisme bootstrap dan random feature selection.

3. Penggunaan **Stratified Sampling** pada Table Partitioner memastikan bahwa proporsi Male dan Female tetap terjaga di kedua partisi, sehingga evaluasi model dilakukan secara fair.
