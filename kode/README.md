# kode/ — Cara Menjalankan dan Mereproduksi

Folder ini berisi seluruh kode untuk UTS Deep Learning (Task 1 klasifikasi, Task 2 regresi RUL, Task 3 clustering). Satu perintah mereproduksi seluruh angka di `laporan.pdf`.

## 1. Persiapan

```bash
git clone https://github.com/sebastianvettel1212/UTS-Deep-Learning.git
cd UTS-Deep-Learning
pip install -r kode/requirements.txt
```

Unduh `data_uts_dl.zip` dari [Google Drive](https://drive.google.com/drive/folders/15TiPGfsw0Irp36QM_zwd_oZnfcQquQ9x?usp=sharing), lalu ekstrak ke folder `data/` di root repositori (data tidak di-commit). Definisi tiap file ada di `kamus_data.md`.

## 2. Satu perintah reproduksi

```bash
bash kode/run_all.sh
```
Atau dapat dieksekusi langsung via Python:
```bash
python kode/run_all.py --seed 42
```

Perintah ini menjalankan ketiga task secara berurutan dan menghasilkan:

| Keluaran                   | Task | Format & Keterangan |
| -------------------------- | ---- | ------------------- |
| `prediksi_klasifikasi.csv` | 1    | 16.000 baris (`id_unit,kelas`) |
| `prediksi_rul.csv`         | 2    | 300 baris (`id_unit,rul`, rentang 0–150) |
| `cluster.csv`              | 3    | 36.000 baris (`id_sampel,cluster`, integer $K=5$) |

## 3. Isi folder

| File / folder                | Isi                                                        | Penanggung jawab |
| ---------------------------- | ---------------------------------------------------------- | ---------------- |
| `demo_task1.ipynb`           | Task 1: klasifikasi multimodal (gambar + sinyal)           | Zacky            |
| `demo_task2.ipynb`           | Task 2: regresi RUL                                        | Alif             |
| `demo_task3.ipynb`           | Task 3: clustering (tahap a–c Zacky, tahap d–f Alif)       | Zacky dan Alif   |
| `run_all.sh`                 | Menjalankan seluruh pipeline (wrapper shell)               | Bersama          |
| `run_all.py`                 | Master pipeline eksekusi end-to-end Python                 | Bersama          |
| `models/`                    | Bobot model terlatih (.pt, .pkl) untuk tiap task           | Bersama          |
| `laporan/`                   | Figures laporan, skrip generate_report.py, dan PDF         | Bersama          |
| `Task1_Klasifikasi/`         | Modul arsitektur GMU, training, dan inferensi Task 1       | Zacky            |
| `Task2_RUL/`                 | Modul sliding window, BiLSTM-Attention, inferensi Task 2  | Alif             |
| `Task3_Klasterisasi/`        | Modul Autoencoder, DEC clustering, dan analisis manifold   | Zacky dan Alif   |
| `utils/`                     | Modul data loader dan perhitungan metrik evaluasi          | Bersama          |
| `requirements.txt`           | Versi library dependensi                                   | Bersama          |

## 4. Seed, versi, dan lingkungan

- **Seed**: `42` (diatur secara konsisten di satu tempat dan dipakai di seluruh tahapan pelatihan dan evaluasi). Hasil multi-seed Task 1 memakai seed: `42, 43, 44` (Macro-F1 rata-rata: $0.9124 \pm 0.0031$).
- **Versi library**: Python `3.10+ / 3.11`, PyTorch `2.2+` (lihat `kode/requirements.txt`).
- **Lingkungan**: Mesin Lokal (Windows / Linux) / Google Colab / Kaggle dengan GPU NVIDIA CUDA atau CPU fallback.

## 5. Bobot model akhir

- Bobot model akhir untuk tiap task telah disertakan langsung di direktori `models/`:
  - `models/task1_best_model.pt` (Task 1 Gated Multimodal Unit)
  - `models/task2_best_model.pt` & `models/task2_scaler.pkl` (Task 2 BiLSTM-Attention)
  - `models/task3_multimodal_ae.pt` (Task 3 Joint Multimodal Autoencoder)
- Seluruh ukuran file bobot berada di bawah ambang batas GitHub (< 100 MB), sehingga dapat diakses dan langsung dimuat tanpa unduhan eksternal tambahan.
- Untuk melatih ulang model dari awal (*retrain*):
  ```bash
  python kode/run_all.py --seed 42
  ```

## 6. Pretrained weights dan sumber

Model dilatih secara *self-supervised* (Autoencoder Task 3) dan *supervised end-to-end* (Task 1 & Task 2) langsung menggunakan dataset resmi tugas UTS Deep Learning. Tidak ada pretrained weights eksternal pihak ketiga atau data proprietary di luar dataset tugas yang digunakan.
