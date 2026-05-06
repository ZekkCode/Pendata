# Tugas Decision Tree C4.5 (Gain Ratio)

Pada tugas ini, saya mendemonstrasikan proses pembuatan model Decision Tree C4.5 menggunakan aplikasi KNIME. Dataset yang digunakan adalah dataset Play Tennis. Penjelasan untuk masing-masing tahap dan komponen di dalam workflow KNIME dijelaskan secara berurutan di bawah.

## Raw Data Perhitungan C4.5 Tennis

Data awal bersumber dari file Excel yang berisi histori kondisi cuaca dan keputusan bermain tenis. Di dalam dataset ini terdapat 14 baris rekaman dengan atribut prediktor: Outlook, Temp., Humidity, dan Wind, serta satu atribut target yaitu Play Tennis.

| # | Outlook | Temp. | Humidity | Wind | Play Tennis |
|---:|:---|:---|:---|:---|:---|
| 0 | Sunny | Hot | High | False | No |
| 1 | Sunny | Hot | High | True | No |
| 2 | Overcast | Hot | High | False | Yes |
| 3 | Rain | Mild | High | False | Yes |
| 4 | Rain | Cool | Normal | False | Yes |
| 5 | Rain | Cool | Normal | True | No |
| 6 | Overcast | Cool | Normal | True | Yes |
| 7 | Sunny | Mild | High | False | No |
| 8 | Sunny | Cold | Normal | False | Yes |
| 9 | Rain | Mild | Normal | False | Yes |
| 10 | Sunny | Mild | Normal | True | Yes |
| 11 | Overcast | Mild | High | True | Yes |
| 12 | Overcast | Hot | Normal | False | Yes |
| 13 | Rain | Mild | High | True | No |

## Impor Data ke KNIME

![Data Raw](Assets/Tugas/TugasDecisionTreeGainRatio/DataRaw.png)

Pada KNIME, saya menggunakan node **Excel Reader** untuk mengimpor file data tersebut. Tampilan di atas adalah hasil pembacaan data. Saya mengecek kembali tipe datanya dan terlihat bahwa semua kolom bertipe String, yang memang sesuai mengingat seluruh variabelnya merupakan data kategorikal.

## Pembagian Data dengan Table Partitioner

![Table Partitioner](Assets/Tugas/TugasDecisionTreeGainRatio/TablePartioner.png)

Untuk melatih dan menguji model, saya perlu membagi data. Saya menggunakan node **Table Partitioner** untuk memecah dataset mentah menjadi dua bagian terpisah: data latih (training set) dan data uji (test set). Data latih nantinya akan digunakan oleh algoritma untuk belajar, sedangkan data uji saya pakai untuk mengukur seberapa baik performa model yang sudah jadi.

## Pembelajaran Model dengan Decision Tree Learner

![Decision Tree Learner](Assets/Tugas/TugasDecisionTreeGainRatio/DecisonTreeLearner.png)

Data latih kemudian saya hubungkan ke node **Decision Tree Learner**. Di dalam node ini, model membaca pola dari atribut input dan secara rekursif memilih atribut dengan metrik pemisahan terbaik (misalnya metrik Gain Ratio pada algoritma C4.5) untuk membagi data. Hasil akhir dari pemrosesan node ini berupa model pohon keputusan.

## Visualisasi Pohon Keputusan

![Decision Tree View](Assets/Tugas/TugasDecisionTreeGainRatio/DecisionTreeView.png)

Untuk mengecek dan memahami logika yang dibuat oleh model, saya menambahkan node **Decision Tree View**. Node ini menerima output model yang sudah dilatih dan menampilkan grafis struktur pohon keputusannya. Melalui visualisasi interaktif tersebut, saya bisa melakukan penelusuran untuk melihat alasan pemilihan setiap percabangan dari akar (root) sampai ke daun (leaf).

## Prediksi menggunakan Decision Tree Predictor

![Decision Tree Predictor](Assets/Tugas/TugasDecisionTreeGainRatio/DecisionTreePredictor.png)

Setelah model selesai dilatih, langkah selanjutnya adalah menerapkannya ke data uji. Saya memakai node **Decision Tree Predictor** dengan memasukkan dua input: model pohon keputusan dan partisi data uji. Node ini akan mencocokkan setiap baris data uji ke dalam aturan model untuk menghasilkan sebuah prediksi baru.

## Hasil Prediksi

![Hasil Decision Tree](Assets/Tugas/TugasDecisionTreeGainRatio/HasilDecisionTree.png)

Tabel di atas menampilkan keluaran dari node predictor. Terdapat penambahan kolom baru di bagian paling kanan bernama "Prediction (Play Tennis)". Dari tabel ini, saya bisa langsung mengobservasi perbandingan antara label target yang sebenarnya (Play Tennis) dengan label hasil tebakan algoritma.

## Evaluasi Akurasi dengan Scorer

![Accuracy Statistics Scorer](Assets/Tugas/TugasDecisionTreeGainRatio/AccuracyStatictsScorer.png)

Sebagai tahap akhir evaluasi, saya melampirkan node **Scorer** untuk mengukur seberapa akurat prediksi yang dihasilkan. Node ini mengambil data hasil prediksi, lalu membandingkan kolom kelas target asli dengan kelas prediksinya. Output akhirnya berupa nilai akurasi secara umum serta confusion matrix yang memudahkan saya mengetahui secara detail jumlah tebakan yang benar dan meleset.
