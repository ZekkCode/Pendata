# Pertemuan 13 - Evaluasi Model

## Tujuan Pembelajaran
```{admonition} Tujuan
:class: tip
1. Mahasiswa dapat memahami metode evaluasi model data mining
2. Mahasiswa dapat menghitung metrik evaluasi (akurasi, presisi, recall, F1-score)
3. Mahasiswa dapat menggunakan confusion matrix
```

---

## Evaluasi Model

### Confusion Matrix

|  | **Prediksi Positif** | **Prediksi Negatif** |
|--|---------------------|---------------------|
| **Aktual Positif** | True Positive (TP) | False Negative (FN) |
| **Aktual Negatif** | False Positive (FP) | True Negative (TN) |

### Metrik Evaluasi

| Metrik | Rumus |
|--------|-------|
| **Akurasi** | $\frac{TP + TN}{TP + TN + FP + FN}$ |
| **Presisi** | $\frac{TP}{TP + FP}$ |
| **Recall** | $\frac{TP}{TP + FN}$ |
| **F1-Score** | $\frac{2 \times Presisi \times Recall}{Presisi + Recall}$ |

### Metode Validasi
- **Hold-out** (Train/Test Split)
- **K-Fold Cross Validation**
- **Leave-One-Out**

---

## Referensi
- [Evaluasi - Situs Dosen](https://mulaab.github.io/datamining/compliance/)

---

## Catatan Perkuliahan

```{note}
Tambahkan catatan perkuliahan Anda di sini setelah pertemuan berlangsung.
```
