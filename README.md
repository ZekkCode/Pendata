<div align="center">

# Student Performance Data Mining

**Academic Project · Data Mining · Semester 4**

[![Live Documentation](https://img.shields.io/badge/Live_Documentation-GitHub_Pages-222222?style=flat-square&logo=github)](https://zekkcode.github.io/Pendata/)
![Python](https://img.shields.io/badge/Python-Data_Analysis-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter Book](https://img.shields.io/badge/Jupyter_Book-Documentation-F37626?style=flat-square&logo=jupyter&logoColor=white)

</div>

## Tentang Project

Repository ini berisi project akademik Mata Kuliah **Penambangan Data** pada Program Studi Teknik Informatika, Universitas Trunojoyo Madura.

Project berfokus pada proses pengolahan dan klasifikasi performa mahasiswa menggunakan dataset **Higher Education Students Performance Evaluation** dari UCI Machine Learning Repository. Hasil performa dikelompokkan menjadi tiga kategori, yaitu **Rendah, Sedang, dan Tinggi**.

Dokumentasi materi, eksperimen, dan hasil analisis disusun menggunakan Jupyter Book agar proses pengerjaan dapat dibaca secara runtut melalui website.

## Tujuan Project

- Menyiapkan dan memahami dataset performa mahasiswa.
- Melakukan preprocessing serta eksplorasi data.
- Membangun model klasifikasi performa mahasiswa.
- Membandingkan pendekatan **Decision Tree** dan **Random Forest**.
- Membandingkan workflow analisis menggunakan Python, Orange, dan KNIME.
- Menyajikan proses serta hasil analisis dalam dokumentasi interaktif.

## Kontribusi Saya

- Menyiapkan dan membersihkan data.
- Menentukan target klasifikasi Rendah, Sedang, dan Tinggi.
- Melakukan eksperimen serta perbandingan model.
- Menyusun workflow analisis pada Python, Orange, dan KNIME.
- Membuat dokumentasi project menggunakan Jupyter Book.
- Mempublikasikan dokumentasi melalui GitHub Pages.

## Teknologi dan Tools

| Kategori | Teknologi |
|---|---|
| Bahasa | Python |
| Analisis Data | Pandas, NumPy |
| Machine Learning | Decision Tree, Random Forest |
| Notebook | Jupyter Notebook |
| Visual Workflow | Orange, KNIME |
| Dokumentasi | Jupyter Book |
| Deployment | GitHub Pages |

## Dokumentasi Online

Dokumentasi project dapat dibaca melalui:

**https://zekkcode.github.io/Pendata/**

## Menjalankan Dokumentasi Secara Lokal

### 1. Aktifkan environment

```powershell
..\.venv\Scripts\Activate.ps1
```

Apabila PowerShell membatasi eksekusi script:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
```

### 2. Instal dependensi dokumentasi

```bash
pip install "jupyter-book==1.0.0" ghp-import
```

### 3. Build Jupyter Book

```bash
jupyter-book build materi-pendat
```

Hasil build tersedia di:

```text
materi-pendat/_build/html/index.html
```

## Deployment GitHub Pages

```bash
ghp-import -n -p -f materi-pendat/_build/html
```

Keterangan:

- `-n` membuat file `.nojekyll`.
- `-p` langsung melakukan push.
- `-f` menimpa branch deployment sebelumnya.

## Konteks Akademik

- **Mata Kuliah:** Penambangan Data
- **Program Studi:** Teknik Informatika
- **Universitas:** Universitas Trunojoyo Madura
- **Pengembang:** Zakaria Mujur Prasetyo

## Catatan

Repository ini dibuat untuk kebutuhan pembelajaran dan evaluasi akademik. Dataset, model, serta hasil eksperimen digunakan sebagai bahan analisis dan tidak ditujukan sebagai sistem penilaian mahasiswa secara nyata.
