# Kontribusi Anggota dan Penggunaan AI

UTS Deep Learning, tugas kelompok. Repositori kelompok: [https://github.com/sebastianvettel1212/UTS-Deep-Learning](https://github.com/sebastianvettel1212/UTS-Deep-Learning)

## 1. Anggota dan peran

| Anggota              | NIM          | Akun GitHub                                                   | Peran                                                                                                                                                  |
| -------------------- | ------------ | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Muhammad Alif Naufal | 101032300098 | [alifnaufalcs](https://github.com/alifnaufalcs)               | Task 2 (regresi RUL) seluruhnya dan `prediksi_rul.csv`; Task 3 tahap a–c (eksplorasi distorsi, representasi unsupervised SimCLR/autoencoder, fusi dua modalitas); draf laporan Task 2 & Task 3 a–c |
| Zacky Yusup Hakim    | 101032300183 | [sebastianvettel1212](https://github.com/sebastianvettel1212) | Task 1 (klasifikasi multimodal) seluruhnya dan `prediksi_klasifikasi.csv`; Task 3 tahap d–f (clustering, penentuan jumlah cluster K=5, analisis manifold t-SNE/UMAP) dan `cluster.csv`; pemilik repositori; draf laporan Task 1 & Task 3 d–f |

Task 3 dibagi dua: Alif mengerjakan tahap a–c, Zacky mengerjakan tahap d–f. Laporan, `kode/README.md`, dan satu perintah reproduksi disusun bersama dan diperiksa oleh kedua anggota.

### Rincian pekerjaan

**Muhammad Alif Naufal: Task 2** (`kode/02_regresi.ipynb`, `prediksi_rul.csv`)

- Eksplorasi dan praproses: pengaruh operating settings terhadap sensor, eliminasi sensor konstan / tidak informatif, missing value handling (forward-fill), deteksi spike, dan z-score normalization.
- Pembuatan label RUL piecewise linear dari riwayat run-to-failure dengan batas atas (clipping) 150 siklus.
- Sliding window (panjang sekuens $W=30$) untuk riwayat dengan panjang berbeda tanpa kebocoran informasi masa depan.
- Perbandingan baseline non-deep-learning (Random Forest & Ridge) dan arsitektur sekuens deep learning (BiLSTM dengan Temporal Self-Attention).
- Validasi dengan split berbasis ID unit mesin (bukan per baris); evaluasi metrik RMSE dan skor asimetris per rentang degradasi RUL.
- Estimasi ketidakpastian interval prediksi menggunakan Monte Carlo (MC) Dropout serta perumusan rekomendasi preventif kapan mesin sebaiknya diservis sebelum batas kritis.

**Muhammad Alif Naufal: Task 3 tahap a–c** (`kode/03_clustering.ipynb`, sel tahap a s.d. c)

- Eksplorasi distorsi citra mikroskop (perbedaan intensitas dan kontras mikroskop M1 vs M2, rotasi, skala, pencahayaan) serta distorsi spektrum (baseline drift, peak shift, spike noise pada 22.15% sampel).
- Pembelajaran representasi tanpa label (*unsupervised representation learning*): augmentasi domain-spesifik dan pelatihan contrastive learning (SimCLR multimodal) serta joint autoencoder untuk mereduksi citra dan spektrum ke dalam ruang laten kompak.
- Evaluasi representasi gabungan modalitas citra dan spektral (*multimodal fusion*) serta analisis mendalam mengenai apa yang hilang jika hanya menggunakan modalitas tunggal (*image-only* vs *spectrum-only*).

**Zacky Yusup Hakim: Task 1** (`kode/01_Classification.ipynb`, `prediksi_klasifikasi.csv`)

- Eksplorasi data: analisis visual contoh sampel 7 kelas pada citra dan sensor tekan, distribusi frekuensi kelas, analisis missing sensor modality (MCAR ~5%), serta pergeseran domain antar lini produksi (Lini A vs Lini B).
- Perancangan arsitektur: ResNet visual encoder, 1D-CNN/BiGRU sensor encoder, dan perbandingan strategi fusi (Early Concatenation vs Late Fusion vs Cross-Attention).
- Eksperimen nilai modalitas tunggal (image-only vs sensor-only), mitigasi label noise dengan Label Smoothing, penanganan pergeseran domain per lini, serta modality dropout saat sensor tidak terekam.
- Sanity check overfit satu batch kecil, analisis kurva konvergensi pelatihan, dan evaluasi performa multi-seed (seed 42, 142, 242) untuk memastikan kestabilan Macro-F1 model akhir.

**Zacky Yusup Hakim: Task 3 tahap d–f** (`kode/03_clustering.ipynb`, sel tahap d s.d. f; `cluster.csv` / `prediksi_cluster.csv`)

- Perbandingan metode clustering pada representasi gabungan embedding multimodal: K-Means++, Gaussian Mixture Models (GMM), Agglomerative Clustering, dan Deep Embedded Clustering (DEC) dengan Student's t-distribution soft assignment.
- Penentuan jumlah cluster optimal ($K=5$) berdasarkan evaluasi multi-kriteria independen: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, GMM Information Criteria (BIC & AIC), dan stabilitas multi-seed (ARI).
- Visualisasi manifold 2D interaktif (UMAP dan t-SNE), verifikasi invariansi domain mikroskop M1 vs M2, serta analisis komprehensif mengenai cluster material yang memiliki batas separasi samar akibat irisan spektrum komposisi kimia.

## 2. Pull request dan jejak commit

| Pekerjaan                                  | Penulis              | Tautan                                                                                                   |
| ------------------------------------------ | -------------------- | -------------------------------------------------------------------------------------------------------- |
| Task 2 (regresi RUL) dan `prediksi_rul.csv` | Muhammad Alif Naufal | [PR #2](https://github.com/sebastianvettel1212/UTS-Deep-Learning/pull/2)                                 |
| Task 3 tahap a–c                            | Muhammad Alif Naufal | [PR #2](https://github.com/sebastianvettel1212/UTS-Deep-Learning/pull/2)                                 |
| Task 1 dan `prediksi_klasifikasi.csv`       | Zacky Yusup Hakim    | [Commit main (Repo Owner)](https://github.com/sebastianvettel1212/UTS-Deep-Learning/commits/main)       |
| Task 3 tahap d–f dan `cluster.csv`          | Zacky Yusup Hakim    | [Commit main (Repo Owner)](https://github.com/sebastianvettel1212/UTS-Deep-Learning/commits/main)       |
| Laporan, README, satu perintah reproduksi   | Bersama (Alif & Zacky) | [PR #2 & Main Commits](https://github.com/sebastianvettel1212/UTS-Deep-Learning/pull/2)                 |

Catatan: Zacky adalah pemilik repositori sehingga pekerjaannya diajukan langsung melalui branch `main` repositori. Alif bekerja di fork [https://github.com/alifnaufalcs/UTS-Deep-Learning](https://github.com/alifnaufalcs/UTS-Deep-Learning) dan perubahan digabungkan melalui Pull Request yang telah di-merge.

## 3. Penggunaan alat AI

| Alat                | Dipakai oleh         | Bagian yang dibantu                                                                                                  |
| ------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Claude (Anthropic) / Antigravity AI | Muhammad Alif Naufal | Perancangan arsitektur BiLSTM-Attention Task 2, kalibrasi MC Dropout, formulasi loss asimetris, eksplorasi distorsi mikroskop & representasi SimCLR Task 3 tahap a–c, serta penyusunan dokumentasi |
| Antigravity AI / GitHub Copilot     | Zacky Yusup Hakim    | Kerangka pipeline klasifikasi multimodal Task 1 (ResNet + 1D-CNN), benchmarking DEC multimodal Task 3 tahap d–f, dan skrip visualisasi |

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
