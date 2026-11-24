# Pipeline Deep Learning - Task 1: Multimodal Defect Classification

## Cara Menjalankan (Reproducibility)
1. **Smoke Test Cepat (--quick)**:
   ```bash
   python run_all.py --quick
   ```
2. **Full Training & Evaluasi**:
   ```bash
   python run_all.py
   ```
3. **Full 3-Seed Ensemble (Task g)**:
   ```bash
   python run_all.py --run_3seeds
   ```

## Struktur File
- `configs/`: Konfigurasi hyperparameter
- `src/data.py`: Pipeline dataset, augmentasi, handling sensor NaN, modality dropout
- `src/models.py`: ImageEncoder (ResNet2D), SignalEncoder (ResNet1D), Concat, LateFusion, CrossAttention
- `src/losses.py`: Weighted CrossEntropy, Label Smoothing, Soft Bootstrapping Loss
- `src/train.py`: Engine training dengan mixed precision autocast, scheduler, sanity check
- `src/eval.py`: Evaluasi modul a-g, domain shift, sensor masking, test prediction & validator
- `src/eda.py`: Visualisasi EDA multimodal, missing rate, drift analysis
- `run_all.py`: CLI satu perintah reproduksi seluruh laporan

## Versi Pustaka Utama (Tercatat di requirements.txt)
- Python 3.12.13
- PyTorch: 2.6.0+cu124
- Torchvision: 0.21.0+cu124
- NumPy: 2.5.4
- Pandas: 3.1.1
- Scikit-Learn: 1.9.1
- Matplotlib: 3.11.2
- Seaborn: 0.13.2

## Random Seed
- Seed Default: 42
- Seeds Ensemble: [42, 142, 242]

## Sumber Pretrained Weights
- Seluruh arsitektur dioptimalkan secara *from scratch* dan disesuaikan secara presisi untuk resolusi citra 64x64 dan sinyal 128-timestep agar hemat komputasi (bebas ketergantungan model eksternal dan bebas resiko copyright/domain mismatch ImageNet).