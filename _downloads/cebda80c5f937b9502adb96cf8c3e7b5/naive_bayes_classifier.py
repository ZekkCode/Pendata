"""
==============================================================================
Analisa Data Menggunakan Naive Bayes Classifier
==============================================================================
Nama    : Zakaria Mujur Prasetyo
NIM     : 240411100144
Kelas   : Penambangan Data A
Dosen   : Mula'ab, S.Si., M.Kom

Deskripsi:
    Script ini melakukan klasifikasi menggunakan algoritma Naive Bayes
    (GaussianNB) dari library scikit-learn pada dataset Iris.
    
    Tahapan:
    1. Load dataset Iris dari sklearn
    2. Eksplorasi dan visualisasi data
    3. Preprocessing (split data train/test)
    4. Training model Gaussian Naive Bayes
    5. Prediksi dan evaluasi model
    6. Visualisasi hasil (Confusion Matrix, distribusi fitur)

Referensi:
    - https://scikit-learn.org/stable/api/sklearn.naive_bayes.html
    - https://scikit-learn.org/stable/modules/naive_bayes.html
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
import os

# ============================================================================
# Konfigurasi output gambar
# ============================================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "Assets", "NaiveBayes")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("=" * 70)
print("ANALISA DATA MENGGUNAKAN NAIVE BAYES CLASSIFIER")
print("Dataset: Iris (sklearn)")
print("=" * 70)

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["species"] = iris.target
df["species_name"] = df["species"].map(
    {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
)

# ============================================================================
# 2. EKSPLORASI DATA
# ============================================================================
print("\n" + "=" * 70)
print("EKSPLORASI DATA")
print("=" * 70)

print("\n--- Informasi Dataset ---")
print(f"Jumlah sampel  : {df.shape[0]}")
print(f"Jumlah fitur   : {df.shape[1] - 2}")  # minus species & species_name
print(f"Jumlah kelas   : {df['species'].nunique()}")
print(f"Nama kelas     : {', '.join(df['species_name'].unique())}")

print("\n--- 5 Data Pertama ---")
print(df.head().to_string(index=False))

print("\n--- Statistik Deskriptif ---")
print(df.describe().to_string())

print("\n--- Distribusi Kelas ---")
distribusi = df["species_name"].value_counts()
for kelas, jumlah in distribusi.items():
    print(f"  {kelas:12s} : {jumlah} sampel ({jumlah/len(df)*100:.1f}%)")
print(f"  {'Total':12s} : {len(df)} sampel")

print("\n--- Missing Values ---")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  Tidak ada missing values pada dataset ini.")
else:
    print(missing[missing > 0])

# ============================================================================
# 3. VISUALISASI DATA
# ============================================================================
print("\n" + "=" * 70)
print("VISUALISASI DATA")
print("=" * 70)

# --- 3a. Distribusi setiap fitur per kelas ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Distribusi Fitur Dataset Iris per Kelas", fontsize=16, fontweight="bold")

colors = {"Setosa": "#2ecc71", "Versicolor": "#3498db", "Virginica": "#e74c3c"}

for idx, col in enumerate(iris.feature_names):
    ax = axes[idx // 2, idx % 2]
    for species_name, color in colors.items():
        subset = df[df["species_name"] == species_name]
        ax.hist(subset[col], bins=15, alpha=0.6, label=species_name, color=color, edgecolor="white")
    ax.set_title(col.replace(" (cm)", "").title(), fontsize=12, fontweight="bold")
    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel("Frekuensi", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "distribusi_fitur.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] Gambar distribusi fitur disimpan.")

# --- 3b. Scatter plot pasangan fitur utama ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Scatter Plot Fitur Iris", fontsize=14, fontweight="bold")

for species_name, color in colors.items():
    subset = df[df["species_name"] == species_name]
    axes[0].scatter(
        subset["sepal length (cm)"], subset["sepal width (cm)"],
        c=color, label=species_name, alpha=0.7, edgecolors="white", s=60
    )
    axes[1].scatter(
        subset["petal length (cm)"], subset["petal width (cm)"],
        c=color, label=species_name, alpha=0.7, edgecolors="white", s=60
    )

axes[0].set_xlabel("Sepal Length (cm)")
axes[0].set_ylabel("Sepal Width (cm)")
axes[0].set_title("Sepal Length vs Sepal Width")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_xlabel("Petal Length (cm)")
axes[1].set_ylabel("Petal Width (cm)")
axes[1].set_title("Petal Length vs Petal Width")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scatter_plot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] Gambar scatter plot disimpan.")

# --- 3c. Heatmap korelasi ---
fig, ax = plt.subplots(figsize=(8, 6))
correlation = df[iris.feature_names].corr()
sns.heatmap(
    correlation, annot=True, fmt=".2f", cmap="RdYlBu_r",
    linewidths=0.5, ax=ax, vmin=-1, vmax=1,
    square=True, cbar_kws={"shrink": 0.8}
)
ax.set_title("Matriks Korelasi Fitur Iris", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "korelasi_fitur.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] Gambar korelasi fitur disimpan.")

# ============================================================================
# 4. PREPROCESSING — Split Data
# ============================================================================
print("\n" + "=" * 70)
print("PREPROCESSING — SPLIT DATA")
print("=" * 70)

X = df[iris.feature_names].values
y = df["species"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Rasio split       : 80% training, 20% testing")
print(f"  Data training     : {X_train.shape[0]} sampel")
print(f"  Data testing      : {X_test.shape[0]} sampel")

# Distribusi kelas di training dan testing
train_dist = pd.Series(y_train).value_counts().sort_index()
test_dist = pd.Series(y_test).value_counts().sort_index()
label_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

print("\n  Distribusi kelas (training):")
for idx, count in train_dist.items():
    print(f"    {label_map[idx]:12s} : {count} sampel")

print("\n  Distribusi kelas (testing):")
for idx, count in test_dist.items():
    print(f"    {label_map[idx]:12s} : {count} sampel")

# ============================================================================
# 5. TRAINING MODEL — Gaussian Naive Bayes
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING MODEL — GAUSSIAN NAIVE BAYES")
print("=" * 70)

model = GaussianNB()
model.fit(X_train, y_train)

print("\n  Model berhasil dilatih!")
print(f"  Algoritma         : Gaussian Naive Bayes (GaussianNB)")
print(f"  Library            : scikit-learn (sklearn.naive_bayes)")
print(f"  Jumlah kelas       : {len(model.classes_)}")
print(f"  Prior probability  :")
for cls, prior in zip(model.classes_, model.class_prior_):
    print(f"    P({label_map[cls]:12s}) = {prior:.4f}")

print(f"\n  Mean (theta) per kelas per fitur:")
for cls_idx, cls in enumerate(model.classes_):
    print(f"\n    Kelas: {label_map[cls]}")
    for feat_idx, feat_name in enumerate(iris.feature_names):
        mean_val = model.theta_[cls_idx, feat_idx]
        var_val = model.var_[cls_idx, feat_idx]
        print(f"      {feat_name:25s} : mu={mean_val:.4f}, sigma²={var_val:.4f}")

# ============================================================================
# 6. PREDIKSI DAN EVALUASI
# ============================================================================
print("\n" + "=" * 70)
print("PREDIKSI DAN EVALUASI MODEL")
print("=" * 70)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# --- Metrik Evaluasi ---
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted")
rec = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\n  --- Metrik Evaluasi (Weighted Average) ---")
print(f"  Accuracy    : {acc:.4f} ({acc*100:.2f}%)")
print(f"  Precision   : {prec:.4f} ({prec*100:.2f}%)")
print(f"  Recall      : {rec:.4f} ({rec*100:.2f}%)")
print(f"  F1-Score    : {f1:.4f} ({f1*100:.2f}%)")

# --- Classification Report ---
print("\n  --- Classification Report ---")
report = classification_report(
    y_test, y_pred,
    target_names=["Setosa", "Versicolor", "Virginica"]
)
print(report)

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
print("  --- Confusion Matrix ---")
print(f"  {'':15s} {'Pred Setosa':>12s} {'Pred Versicolor':>16s} {'Pred Virginica':>15s}")
for i, cls_name in enumerate(["Setosa", "Versicolor", "Virginica"]):
    print(f"  {cls_name:15s} {cm[i, 0]:>12d} {cm[i, 1]:>16d} {cm[i, 2]:>15d}")

# --- Perhitungan manual per kelas ---
print("\n  --- Perhitungan Metrik per Kelas (Manual) ---")
for i, cls_name in enumerate(["Setosa", "Versicolor", "Virginica"]):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    tn = cm.sum() - tp - fp - fn

    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
    a = (tp + tn) / (tp + tn + fp + fn)

    print(f"\n  Kelas: {cls_name}")
    print(f"    TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}")
    print(f"    Precision = TP/(TP+FP) = {tp}/({tp}+{fp}) = {p:.4f}")
    print(f"    Recall    = TP/(TP+FN) = {tp}/({tp}+{fn}) = {r:.4f}")
    print(f"    F1-Score  = 2*P*R/(P+R) = 2*{p:.4f}*{r:.4f}/({p:.4f}+{r:.4f}) = {f:.4f}")
    print(f"    Accuracy  = (TP+TN)/(TP+TN+FP+FN) = ({tp}+{tn})/({tp}+{tn}+{fp}+{fn}) = {a:.4f}")

# ============================================================================
# 7. VISUALISASI HASIL
# ============================================================================
print("\n" + "=" * 70)
print("VISUALISASI HASIL")
print("=" * 70)

# --- 7a. Confusion Matrix Heatmap ---
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Setosa", "Versicolor", "Virginica"],
    yticklabels=["Setosa", "Versicolor", "Virginica"],
    linewidths=1, linecolor="white",
    annot_kws={"size": 16, "fontweight": "bold"},
    ax=ax
)
ax.set_title("Confusion Matrix — Gaussian Naive Bayes", fontsize=14, fontweight="bold")
ax.set_xlabel("Prediksi", fontsize=12)
ax.set_ylabel("Aktual", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] Confusion matrix disimpan.")

# --- 7b. Bar chart metrik evaluasi ---
fig, ax = plt.subplots(figsize=(8, 5))
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
values = [acc, prec, rec, f1]
bar_colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]

bars = ax.bar(metrics, values, color=bar_colors, edgecolor="white", width=0.6)
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
        f"{val:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold"
    )

ax.set_ylim(0, 1.15)
ax.set_ylabel("Nilai", fontsize=12)
ax.set_title("Metrik Evaluasi Model Naive Bayes", fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metrik_evaluasi.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] Bar chart metrik evaluasi disimpan.")

# --- 7c. Probabilitas prediksi untuk beberapa sampel ---
fig, ax = plt.subplots(figsize=(12, 5))
sample_indices = [0, 5, 10, 15, 20, 25]
sample_probs = y_pred_proba[sample_indices]
x_pos = np.arange(len(sample_indices))
width = 0.25

for i, (cls_name, color) in enumerate(colors.items()):
    ax.bar(x_pos + i * width, sample_probs[:, i], width,
           label=cls_name, color=color, edgecolor="white")

ax.set_xlabel("Sampel ke-", fontsize=12)
ax.set_ylabel("Probabilitas", fontsize=12)
ax.set_title("Probabilitas Prediksi Naive Bayes (Sampel Terpilih)", fontsize=14, fontweight="bold")
ax.set_xticks(x_pos + width)
ax.set_xticklabels([f"#{i}" for i in sample_indices])
ax.legend(title="Kelas")
ax.set_ylim(0, 1.1)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "probabilitas_prediksi.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] Probabilitas prediksi disimpan.")

# ============================================================================
# 8. CONTOH PREDIKSI DATA BARU
# ============================================================================
print("\n" + "=" * 70)
print("CONTOH PREDIKSI DATA BARU")
print("=" * 70)

data_baru = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Seharusnya Setosa
    [6.7, 3.1, 4.7, 1.5],  # Seharusnya Versicolor
    [7.7, 2.8, 6.7, 2.0],  # Seharusnya Virginica
])

pred_baru = model.predict(data_baru)
prob_baru = model.predict_proba(data_baru)

print("\n  Data baru yang diprediksi:")
for i, (data, pred, prob) in enumerate(zip(data_baru, pred_baru, prob_baru)):
    print(f"\n  Sampel {i+1}: {data}")
    print(f"    Prediksi         : {label_map[pred]}")
    print(f"    Probabilitas     :")
    for cls_idx, cls_name in label_map.items():
        marker = " <-- PREDIKSI" if cls_idx == pred else ""
        print(f"      P({cls_name:12s}) = {prob[cls_idx]:.6f}{marker}")

# ============================================================================
# 9. RINGKASAN AKHIR
# ============================================================================
print("\n" + "=" * 70)
print("RINGKASAN AKHIR")
print("=" * 70)
print(f"""
  Dataset          : Iris (sklearn.datasets)
  Jumlah data      : {len(df)} sampel
  Jumlah fitur     : 4
  Jumlah kelas     : 3 (Setosa, Versicolor, Virginica)
  Algoritma        : Gaussian Naive Bayes
  Rasio split      : 80:20 (stratified)
  
  Hasil Evaluasi:
    Accuracy       : {acc:.4f} ({acc*100:.2f}%)
    Precision      : {prec:.4f} ({prec*100:.2f}%)
    Recall         : {rec:.4f} ({rec*100:.2f}%)
    F1-Score       : {f1:.4f} ({f1*100:.2f}%)

  Gambar hasil disimpan di: {OUTPUT_DIR}
""")
print("=" * 70)
print("SELESAI")
print("=" * 70)
