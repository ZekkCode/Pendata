# UAS | Analisis Perbandingan Decision Tree dan Random Forest

Evaluasi performa akademik mahasiswa ("Higher Education Students Performance Evaluation") menggunakan algoritma Decision Tree dan Random Forest. 

Analisis komprehensif ini dibagi menjadi dua pendekatan (berbasis *code* dan berbasis *visual*), yang dapat Anda akses melalui daftar isi di bawah ini:


## Ringkasan Dataset & Pra-pemrosesan Data

Sebelum masuk ke tahap analisis dan klasifikasi, berikut adalah ringkasan singkat mengenai dataset dan proses transformasi yang dilakukan secara seragam, baik di Python (Jupyter Notebook) maupun Orange Data Mining:

1. **Jumlah Data:** Dataset ini terdiri dari **145 baris** data mahasiswa. Pada format aslinya, dataset ini memiliki 33 kolom (terdiri dari *Student ID*, *Course ID*, 30 kolom pertanyaan, dan 1 target *Grade*).
2. **Pemilihan 8 Fitur Utama (*Feature Selection*):** Dari 30 atribut pertanyaan yang tersedia, analisis ini mengerucut dan difokuskan pada **8 fitur pertama saja** (Fitur 1 sampai 8). Penyederhanaan dimensi ini bertujuan agar struktur pohon (*Decision Tree*) yang nantinya terbentuk tidak terlalu kompleks, lebih mudah diinterpretasikan secara visual, serta meminimalisir risiko *overfitting*. 
   - **Di Python:** Dilakukan dengan melakukan *slicing* / *subsetting* kolom pada *DataFrame* pandas (`df[['1', '2', ..., '8']]`).
   - **Di Orange:** Dilakukan dengan menggunakan widget **Select Columns** untuk membuang kolom yang tidak perlu dan memisahkan fitur dengan target.
3. **Kategorisasi Label Target:** Target *Grade* bawaan dataset memiliki 8 tingkatan/kelas (angka `0` hingga `7`). Memprediksi 8 kelas hanya dengan 145 baris data akan membuat model sangat tidak stabil dan akurasinya terpecah. Sebagai solusinya, target disederhanakan menjadi 3 kategori performa:
   Pengelompokan target ke dalam 3 level performa ini tidak hanya sekadar untuk menyiasati sedikitnya baris data, tetapi juga sejalan dengan evaluasi capaian kemampuan edukasional yang logis di dunia nyata:
   - **Rendah (Grade 0, 1, dan 2):** Mewakili kelompok mahasiswa yang *underperforming*, berisiko, atau membutuhkan bimbingan akademik yang intensif.
   - **Sedang (Grade 3, 4, dan 5):** Mewakili kelompok mahasiswa dengan kemampuan rata-rata yang sudah memenuhi standar kompetensi dasar kelas.
   - **Tinggi (Grade 6 dan 7):** Mewakili kelompok unggulan dari mahasiswa berprestasi yang menguasai materi pembelajaran secara optimal.
   
   - **Di Python:** Transformasi ini diterapkan menggunakan fungsi Python (kondisional *If-Else*) yang dipetakan (*apply*) ke seluruh baris target.
   - **Di Orange:** Proses konversi kategori ini juga disiapkan di awal aliran data agar klasifikasi (*Test & Score*) berjalan seimbang memprediksi 3 kategori tersebut.

```{tableofcontents}
```
