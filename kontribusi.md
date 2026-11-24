# Kontribusi Anggota dan Penggunaan AI

UTS Deep Learning, tugas kelompok. Repositori kelompok: [https://github.com/sebastianvettel1212/UTS-Deep-Learning](https://github.com/sebastianvettel1212/UTS-Deep-Learning)

## 1. Anggota dan peran

| Anggota              | NIM          | Akun GitHub                                                   | Peran                                                                                                                                                  |
| -------------------- | ------------ | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Muhammad Alif Naufal | 101032300098 | [alifnaufalcs](https://github.com/alifnaufalcs)               | Task 2 (regresi RUL) seluruhnya dan `prediksi_rul.csv`; Task 3 tahap d–f (clustering, jumlah cluster, analisis) dan `cluster.csv`; draf laporan Task 2 |
| Zacky Yusup Hakim    | 101032300183 | [sebastianvettel1212](https://github.com/sebastianvettel1212) | Task 1 (klasifikasi multimodal) seluruhnya dan `prediksi_klasifikasi.csv`; Task 3 tahap a–c (eksplorasi, representasi, penggabungan dua modalitas); pemilik repositori; draf laporan Task 1 |

Task 3 dibagi dua: Zacky mengerjakan tahap a–c, Alif mengerjakan tahap d–f. Laporan, `kode/README.md`, dan satu perintah reproduksi disusun bersama dan diperiksa oleh kedua anggota.

### Rincian pekerjaan

**Muhammad Alif Naufal: Task 2** (`kode/demo_task2.ipynb`, `prediksi_rul.csv`)

- Eksplorasi dan praproses: pengaruh operating settings terhadap sensor, eliminasi sensor konstan / tidak informatif, missing value handling (forward-fill), deteksi spike, dan z-score normalization.
- Pembuatan label RUL piecewise linear dari riwayat run-to-failure dengan batas atas (clipping) 150 siklus.
- Sliding window (panjang sekuens $W=30$) untuk riwayat dengan panjang berbeda tanpa kebocoran informasi masa depan.
- Perbandingan baseline non-deep-learning (Random Forest & Ridge) dan arsitektur sekuens deep learning (BiLSTM dengan Temporal Self-Attention).
- Validasi dengan split berbasis ID unit mesin (bukan per baris); evaluasi metrik RMSE dan skor asimetris per rentang degradasi RUL.
- Estimasi ketidakpastian interval prediksi menggunakan Monte Carlo (MC) Dropout serta perumusan rekomendasi preventif kapan mesin sebaiknya diservis sebelum batas kritis.

**Muhammad Alif Naufal: Task 3 tahap d–f** (`kode/demo_task3.ipynb`, sel setelah tahap c; `cluster.csv`)

- Perbandingan metode clustering pada representasi gabungan embedding multimodal: K-Means, Gaussian Mixture Models (GMM), dan Deep Embedded Clustering (DEC) dengan Student's t-distribution soft assignment.
- Penentuan jumlah cluster optimal ($K=5$) berdasarkan evaluasi multi-kriteria: Silhouette Score, Davies-Bouldin Index, dan Calinski-Harabasz Index.
- Visualisasi manifold 2D interaktif (UMAP dan t-SNE) serta analisis komprehensif mengenai cluster material yang memiliki batas separasi samar akibat noise komposisi spektral.

**Zacky Yusup Hakim: Task 1** (`kode/01_Classification.ipynb`, `prediksi_klasifikasi.csv`)

- Eksplorasi data: analisis visual contoh sampel 7 kelas pada citra dan sensor tekan, distribusi frekuensi kelas, analisis missing sensor modality, serta variasi distribusi per lini produksi.
- Perancangan arsitektur: ResNet visual encoder, 1D-CNN/BiGRU sensor encoder, dan perbandingan strategi fusi (Early Concatenation vs Late Fusion vs Gated Multimodal Unit).
- Eksperimen nilai modalitas tunggal (image-only vs sensor-only), mitigasi label noise dengan Label Smoothing, normalisasi per lini untuk mengatasi pergeseran domain, serta modality dropout saat sensor tidak terekam.
- Sanity check overfit satu batch kecil, analisis kurva konvergensi pelatihan, dan evaluasi performa multi-seed (seed 42, 43, 44) untuk memastikan kestabilan Macro-F1 model akhir.

**Zacky Yusup Hakim: Task 3 tahap a–c** (`kode/demo_task3.ipynb`, sel sampai tahap c)

- Eksplorasi distorsi mikroskop (rotasi, variasi iluminasi, skala) dan spektrum (baseline drift, peak shift, spike noise).
- Pembelajaran representasi tanpa label: augmentasi domain-spesifik dan pelatihan Joint Multimodal Autoencoder untuk mereduksi citra dan spektrum ke dalam ruang laten kompak.
- Evaluasi representasi gabungan modalitas citra dan spektral serta verifikasi keunggulan representasi multimodal dibandingkan representasi modalitas tunggal.

## 2. Pull request dan jejak commit

| Pekerjaan                                  | Penulis              | Tautan                                                                                                   |
| ------------------------------------------ | -------------------- | -------------------------------------------------------------------------------------------------------- |
| Task 2 (regresi RUL) dan `prediksi_rul.csv` | Muhammad Alif Naufal | [PR #2](https://github.com/sebastianvettel1212/UTS-Deep-Learning/pull/2)                                 |
| Task 3 tahap d–f dan `cluster.csv`          | Muhammad Alif Naufal | [PR #2](https://github.com/sebastianvettel1212/UTS-Deep-Learning/pull/2)                                 |
| Task 1 dan `prediksi_klasifikasi.csv`       | Zacky Yusup Hakim    | [Commit main (Repo Owner)](https://github.com/sebastianvettel1212/UTS-Deep-Learning/commits/main)       |
| Task 3 tahap a–c                            | Zacky Yusup Hakim    | [Commit main (Repo Owner)](https://github.com/sebastianvettel1212/UTS-Deep-Learning/commits/main)       |
| Laporan, README, satu perintah reproduksi   | Bersama (Alif & Zacky) | [PR #2 & Main Commits](https://github.com/sebastianvettel1212/UTS-Deep-Learning/pull/2)                 |

Catatan: Zacky adalah pemilik repositori sehingga pekerjaannya diajukan langsung melalui branch `main` repositori. Alif bekerja di fork [https://github.com/alifnaufalcs/UTS-Deep-Learning](https://github.com/alifnaufalcs/UTS-Deep-Learning) dan perubahan digabungkan melalui Pull Request #2 yang telah berhasil di-merge.

## 3. Penggunaan alat AI

| Alat                | Dipakai oleh         | Bagian yang dibantu                                                                                                  |
| ------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Claude (Anthropic) / Antigravity AI | Muhammad Alif Naufal | Perancangan arsitektur BiLSTM-Attention, formulasi fungsi rugi asimetris teregulasi, implementasi DEC multimodal, serta penyusunan dokumentasi `README.md` dan `kontribusi.md` |
| Antigravity AI / GitHub Copilot     | Zacky Yusup Hakim    | Kerangka pipeline Gated Multimodal Unit (GMU) Task 1, eksplorasi representasi laten Task 3 tahap a–c, dan skrip otomatisasi laporan |

### Cara kelompok memverifikasi keluaran AI

- **Setiap sel dijalankan dan hasilnya dicek** sebelum lanjut; fungsi metrik buatan sendiri (RMSE, skor asimetris, macro-F1, ARI/Silhouette) dibandingkan dengan implementasi pustaka atau perhitungan manual pada contoh kecil.
- **Angka diuji ulang di data asli:** seluruh angka evaluasi di notebook dan laporan dicocokkan dengan hasil menjalankan ulang pipeline dari data mentah.
- **Keputusan dibuktikan dengan angka:** setiap pilihan desain (normalisasi z-score, arsitektur BiLSTM-Attention vs baseline RF, teknik tahan noise label smoothing, penentuan $K=5$) disertai data perbandingan empiris.
- **File prediksi divalidasi:** nama kolom, jumlah baris (16.000 untuk klasifikasi, 300 untuk RUL, 36.000 untuk clustering), keunikan dan kecocokan ID dengan data uji, tidak ada NaN, `rul` di antara 0 dan 150, `kelas` salah satu dari tujuh nama kelas valid, dan `cluster` bilangan bulat non-negatif.
- **Reproduksibilitas:** seluruh angka dapat direproduksi secara konsisten dengan seed tetap (`42`) melalui satu perintah `bash kode/run_all.sh` atau `python kode/run_all.py --seed 42`.
- **Pemahaman:** setiap anggota mempelajari, meninjau silang, dan mampu menjelaskan seluruh alur metodologi dari semua task untuk persiapan ujian lisan.

## 4. Persetujuan anggota

Pernyataan penggunaan AI di atas telah dibaca dan disetujui oleh semua anggota kelompok.

- [x] Muhammad Alif Naufal
- [x] Zacky Yusup Hakim
