# UTS Deep Learning

**Hands-on Deep Learning: Classification, Regression, and Clustering**

Repositori ini berisi pengerjaan UTS (tugas kelompok) mata kuliah Deep Learning, Teknik Komputer, Universitas Telkom. Tugas terdiri dari tiga bagian: klasifikasi multimodal, regresi deret waktu, dan clustering multimodal pada gambar, sinyal sensor, spektrum, dan riwayat mesin.

## Anggota Kelompok

**Kelompok (2 Orang)**

| NIM            | Nama                  |
| -------------- | --------------------- |
| 101032300098   | Muhammad Alif Naufal  |
| 101032300183   | Zacky Yusup Hakim     |

Peran masing-masing anggota, tautan pull request, dan penggunaan AI tools ada di [`kontribusi.md`](kontribusi.md).

## Ringkasan Tugas

| Task | Topik                                               | Target / Output                                                       | Metrik hidden test                          | Bobot |
| ---- | --------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------- | ----- |
| 1    | Klasifikasi multimodal: cacat produk (gambar + sinyal press) | 7 kelas: `normal`, `gores`, `noda`, `retak`, `gelembung`, `las_dingin`, `kontaminasi` | Macro-F1 (7 kelas)                          | 30    |
| 2    | Regresi deret waktu: Remaining Useful Life (RUL)    | RUL pada siklus terakhir tiap mesin uji (dibatasi 0–150)              | RMSE dan skor asimetris                     | 30    |
| 3    | Clustering multimodal: jenis material (36.000 sampel) | Label cluster (jumlah jenis material tidak diberikan)                 | Adjusted Rand Index (ARI)                   | 25    |
| —    | Ujian lisan individual (20 menit)                   | —                                                                     | —                                           | 15    |

### Task 1 — Klasifikasi Multimodal: Product Defects

Setiap unit diperiksa dengan kamera dan sensor tekan. Sebagian cacat hanya tampak di gambar, sebagian hanya di sinyal sensor, dan sebagian samar di keduanya.

- [x] **a. Eksplorasi**: contoh tiap kelas di kedua modalitas, distribusi kelas, rekaman sensor yang hilang, perbedaan train–test (mis. per lini produksi)
- [x] **b. Arsitektur**: image encoder (CNN atau ViT kecil) dan signal encoder (1D-CNN, GRU, atau Transformer); bandingkan minimal dua strategi fusi (early fusion, late fusion, cross-attention)
- [x] **c. Nilai tiap modalitas**: model image-only dan sensor-only, serta kelas mana yang membutuhkan modalitas mana
- [x] **d. Label tidak sempurna**: cek kemungkinan salah label dari inspektur, uji satu teknik yang tahan noise label (mis. label smoothing atau co-teaching)
- [x] **e. Pergeseran domain**: performa per lini produksi pada validasi, uji satu teknik penanganan (mis. augmentasi atau normalisasi per lini)
- [x] **f. Modalitas hilang**: model tetap memberi prediksi wajar saat sensor tidak terekam
- [x] **g. Training**: sanity check (overfit satu batch), kurva belajar, hasil minimal tiga seed untuk model akhir

### Task 2 — Regresi Deret Waktu: Remaining Useful Life

Memprediksi sisa siklus sebelum mesin press gagal, pada siklus terakhir tiap mesin uji. Label RUL dibatasi 150 siklus.

- [x] **a. Eksplorasi dan praproses**: pengaruh operating settings terhadap sensor, sensor tidak informatif, missing value, spike, dan alasan normalisasi
- [x] **b. Label**: label RUL training dari riwayat run-to-failure, termasuk batas 150 siklus
- [x] **c. Sekuens**: sliding window, riwayat dengan panjang berbeda, tanpa kebocoran informasi masa depan
- [x] **d. Model**: minimal dua arsitektur sekuens (LSTM/GRU, 1D-CNN/TCN, atau Transformer) dengan baseline non-deep-learning
- [x] **e. Validasi**: split per mesin (bukan per baris); RMSE dan skor asimetris per rentang RUL
- [x] **f. Ketidakpastian**: interval prediksi (mis. MC dropout atau ensemble) dan diskusi kapan mesin sebaiknya diservis

Skor asimetris pada hidden test, dengan `d = RUL prediksi − RUL aktual`:

```
s = Σ_mesin ( exp(−d/13) − 1   jika d < 0
              exp( d/10) − 1   jika d ≥ 0 )
```

Prediksi yang terlambat (terlalu optimis, `d ≥ 0`) dihukum lebih berat.

### Task 3 — Clustering Multimodal: Material Types

Tiap sampel punya citra mikrostruktur dan spektrum komposisi, tanpa label.

- [x] **a. Eksplorasi**: distorsi gambar (rotasi, skala, kecerahan, perbedaan mikroskop) dan spektrum (shift, baseline, spike)
- [x] **b. Representasi**: representasi tanpa label (autoencoder, contrastive seperti SimCLR/BYOL, atau pretrained embedding) beserta alasan augmentasi
- [x] **c. Dua modalitas**: representasi gambar dan spektrum lalu digabung; tunjukkan apa yang hilang jika hanya satu modalitas
- [x] **d. Clustering**: bandingkan minimal dua metode pada embedding, termasuk satu deep clustering (mis. DEC)
- [x] **e. Jumlah cluster**: minimal dua kriteria
- [x] **f. Analisis**: visualisasi UMAP atau t-SNE; cluster mana yang sulit dipisahkan dan mengapa

## Struktur Repositori (branch `main`)

```
.
├── laporan.pdf                    # Laporan Task 1–3 (maks. 18 halaman)
├── kode/
│   ├── README.md                  # Cara menjalankan dan mereproduksi seluruh angka laporan
│   ├── run_all.py                 # Master pipeline satu perintah reproduksi
│   ├── run_all.sh                 # Shell wrapper untuk run_all.py
│   ├── requirements.txt           # Daftar dependensi Python
│   ├── demo_task1.ipynb           # Notebook demo Task 1
│   ├── demo_task2.ipynb           # Notebook demo Task 2
│   ├── multimodal_clustering_pipeline.ipynb # Notebook utama final Task 3 (Multimodal Clustering)
│   ├── Task1_Klasifikasi/         # Modul klasifikasi multimodal (GMU)
│   ├── Task2_RUL/                 # Modul regresi RUL (BiLSTM-Attention)
│   ├── Task3_Klasterisasi/        # Modul clustering (Autoencoder + DEC)
│   ├── utils/                     # Data loader dan metrik evaluasi
│   ├── models/                    # Bobot model terlatih (.pt, .pkl)
│   └── laporan/                   # Figures laporan, skrip generate_report.py, dan PDF
├── prediksi_klasifikasi.csv       # id_unit, kelas
├── prediksi_rul.csv               # id_unit, rul
├── cluster.csv                    # id_sampel, cluster
└── kontribusi.md                  # Peran anggota, link PR, penggunaan AI tools
```

## Format File Prediksi

Nama file dan nama kolom harus persis sama karena dinilai otomatis.

| File                       | Kolom                | Aturan                                                                  | Contoh baris pertama |
| -------------------------- | -------------------- | ----------------------------------------------------------------------- | -------------------- |
| `prediksi_klasifikasi.csv` | `id_unit,kelas`      | Satu baris per unit uji; `kelas` salah satu dari tujuh nama kelas persis seperti ditulis di atas | `U123456,retak`      |
| `prediksi_rul.csv`         | `id_unit,rul`        | Satu baris per mesin uji; `rul` di antara 0 dan 150                     | `17,87.5`            |
| `cluster.csv`              | `id_sampel,cluster`  | Satu baris per sampel; `cluster` bilangan bulat (nomor boleh bebas)     | `S123456,4`          |

## Dataset

- Data: `data_uts_dl.zip` ([Google Drive](https://drive.google.com/drive/folders/15TiPGfsw0Irp36QM_zwd_oZnfcQquQ9x?usp=sharing)), berisi 100.000 gambar, 100.000 sinyal sensor dan spektrum, riwayat 1.200 mesin, serta `kamus_data.md` (deskripsi tiap file).
- Data **tidak** di-commit ke repositori. Unduh dan ekstrak ke folder `data/` sebelum menjalankan kode.
- Data eksternal tidak diperbolehkan. Pretrained weights yang tersedia untuk umum boleh dipakai dengan menyebutkan sumbernya.

## Reproduksibilitas

Seluruh angka di laporan dapat direproduksi dengan satu perintah. Detail lengkap ada di [`kode/README.md`](kode/README.md).

```bash
# 1. Clone dan masuk ke repo
git clone https://github.com/sebastianvettel1212/UTS-Deep-Learning.git
cd UTS-Deep-Learning

# 2. Install dependensi
pip install -r kode/requirements.txt

# 3. Letakkan data di folder data/, lalu jalankan seluruh pipeline
bash kode/run_all.sh
# Atau eksekusi langsung via Python:
# python kode/run_all.py --seed 42
```

- **Seed**: `42` (diatur di satu tempat dan dipakai di semua eksperimen; hasil multi-seed Task 1 memakai seed: `42, 43, 44`)
- **Versi library**: Python 3.10+ / 3.11, PyTorch >= 2.0 (lihat `kode/requirements.txt`)
- **Lingkungan**: Mesin lokal / Colab / Kaggle (GPU NVIDIA / CPU fallback)
- **Bobot model akhir**: tersedia di `kode/models/` (`task1_best_model.pt`, `task2_best_model.pt`, `task2_scaler.pkl`, `task3_multimodal_ae.pt`)

## Ketentuan Tugas

- **Metode**: model utama setiap task harus deep learning; metode klasik hanya sebagai baseline.
- **Tools**: Python dengan PyTorch atau TensorFlow/Keras.
- **AI tools**: diperbolehkan, dengan keterangan tool yang dipakai, bagian yang dibantu, dan cara verifikasi output. Dicatat di [`kontribusi.md`](kontribusi.md).
- **Kolaborasi antar kelompok**: dilarang berbagi kode, hasil, atau temuan.
- **Tanggung jawab**: setiap anggota harus mampu menjelaskan seluruh pekerjaan kelompok.

## Alur Kerja GitHub

1. Satu anggota membuat repositori kelompok.
2. Setiap anggota **fork** repositori dan bekerja di fork masing-masing.
3. Perubahan diajukan lewat **pull request** ke repositori kelompok, lalu di-merge ke `main`.
4. Sebelum deadline, semua fork di-sync dan semua PR di-merge. Branch `main` pada saat deadline yang dinilai.

Kontribusi dinilai dari commit dan pull request di GitHub.

## Jadwal

| Kegiatan             | Tanggal                                                         |
| -------------------- | --------------------------------------------------------------- |
| Rilis data           | 10 Nov 2026                                                     |
| Deadline pengumpulan | 24 Nov 2026, 23:59 (semua fork ter-sync dan semua PR ter-merge) |
| Ujian lisan          | TBD                                                             |

Keterlambatan: commit setelah deadline dihitung terlambat; nilai kelompok berkurang 10 poin per 24 jam, dan setelah 48 jam tidak diterima.

## Penilaian

**Nilai akhir = nilai kelompok × faktor kontribusi + nilai oral individual**

- Nilai kelompok (Task 1–3): 85 poin; ujian lisan individual: 15 poin.
- Faktor kontribusi: **1.00** (kontribusi terlihat di GitHub dan mampu menjelaskan), **0.85** (kontribusi ada, penjelasan lemah), **0.70** (tidak mampu menjelaskan atau kontribusi minim).
- Ujian lisan: 5 menit reproduksi (satu anggota pilihan dosen menjalankan model akhir dan mereproduksi satu angka laporan secara langsung) dan 15 menit tanya jawab tentang semua task.

## Integritas Akademik

Seluruh pekerjaan dalam repositori ini adalah karya asli kelompok dan mengikuti standar integritas akademik.
