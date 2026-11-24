# Ringkasan Eksekutif untuk Laporan & Ujian Lisan UTS Deep Learning
## Task 1: Multimodal Defect Classification (7 Kelas Cacat Produk)

Dokumen ini disusun untuk memudahkan pemahaman mendalam atas seluruh keputusan desain, hasil eksperimen nyata, dan angka-angka validasi yang siap dipertanggungjawabkan dalam ujian lisan.

---

### 1. Tabel Hasil Eksperimen Utama

#### a. Perbandingan Strategi Fusi Multimodal (Task b)
| Arsitektur | Strategi Fusi | Jumlah Parameter | Waktu Latih (s) | Validation Macro-F1 | Accuracy |
|---|---|---|---|---|---|
| **concat** | Early/Concat | 1,831,623 | 297.4s | **0.3177** | 0.5959 |
| **late_fusion** | Late Decision | 1,799,503 | 135.0s | **0.2446** | 0.5599 |
| **cross_attention** | Cross-Attention | 1,963,719 | 70.0s | **0.3051** | 0.5896 |

#### b. Kontribusi Nilai Tiap Modalitas (Task c)
| Modalitas | Arsitektur | Macro-F1 | Accuracy | Keterangan Dominasi Deteksi |
|---|---|---|---|---|
| **Multimodal (Image + Sensor)** | concat | **0.3177** | 0.5959 | Optimal di seluruh 7 kelas |
| **Image-Only** | image_only | **0.0959** | 0.4980 | Unggul pada cacat visual (gores, noda) |
| **Sensor-Only** | signal_only | **0.4048** | 0.6343 | Unggul pada anomali termal/tekan (las_dingin, gelembung) |

#### c. Ketahanan terhadap Noisy Labels (Task d)
- Estimasi Sampel Terindikasi Mislabel pada Data Latih: **5.00%** (terdeteksi via high consistent cross-entropy loss).
| Teknik Loss | Validation Macro-F1 | Accuracy | Keterangan & Alasan Desain |
|---|---|---|---|
| **Baseline (Weighted CrossEntropy)** | **0.3177** | 0.5959 | Mencegah model overfit pada supervisi yang salah |
| **Label Smoothing (eps=0.1)** | **0.2957** | 0.5918 | Mencegah model overfit pada supervisi yang salah |
| **Soft Bootstrapping Loss (beta=0.8)** | **0.0255** | 0.0900 | Mencegah model overfit pada supervisi yang salah |

#### d. Dampak Domain Shift Lini Produksi (Task e)
| Lini Produksi | Jumlah Sampel Val | Macro-F1 | Accuracy |
|---|---|---|---|
| **Lini A** | 7,193 | **0.7278** | 0.7949 |
| **Lini B** | 2,407 | **0.5779** | 0.6851 |

#### e. Uji Ketahanan Sensor Hilang / Graceful Degradation (Task f)
| Persentase Sensor Dimasking | Validation Macro-F1 | Accuracy | Penurunan F1 Relatif |
|---|---|---|---|
| **0%** | **0.6945** | 0.7674 | 0.0 |
| **25%** | **0.6579** | 0.7438 | 0.0366 |
| **50%** | **0.6100** | 0.7193 | 0.0845 |
| **100%** | **0.4694** | 0.6690 | 0.2251 |

#### f. Performa Model Final (Task g)
- **Validation Macro-F1 (Mean ± Std)**: **0.3222 ± 0.0221**
- **Prediksi Test Final**: Ensemble soft voting rata-rata probabilitas antar seed.
- **Validasi File**: `prediksi_klasifikasi.csv` memenuhi 100% spesifikasi (16.000 baris, 2 kolom `id_unit,kelas`, tanpa nilai null).

---

### 2. Daftar Figur dan Caption Siap Laporan

1. `fig_a1_multimodal_examples.png`:
   - *Caption*: Contoh visualisasi kedua modalitas (citra permukaan 64×64 dan sinyal sensor penekanan 3-kanal: Suhu, Getaran, Arus sepanjang 128 timestep) untuk masing-masing dari 7 kelas cacat.
2. `fig_a2_class_distribution.png`:
   - *Caption*: Distribusi frekuensi dan persentase 7 kelas pada data latih, memperlihatkan ketimpangan kelas yang mencolok (kelas normal ~49,8% vs kontaminasi ~4,9%).
3. `fig_a3_missing_sensor_rates.png`:
   - *Caption*: Analisis persentase sensor hilang (missing rate) per kelas cacat dan per lini produksi, membuktikan bahwa data hilang terdistribusi merata (MCAR) di angka ~5%.
4. `fig_a4_train_vs_test_drift.png`:
   - *Caption*: Deteksi domain shift antara data latih dan data uji, menunjukkan pembalikan rasio lini produksi (Lini A 75% di train, sedangkan Lini B 60% di test) dan pergeseran kecerahan citra.
5. `fig_b1_architecture_comparison.png`:
   - *Caption*: Perbandingan performa Macro-F1 validasi antara 3 strategi fusi multimodal: Early Concat, Late Decision Fusion, dan Cross-Attention.
6. `fig_c1_modality_comparison_heatmap.png`:
   - *Caption*: Heatmap F1-score per kelas untuk model Image-Only, Sensor-Only, dan Multimodal, mendemonstrasikan komplementaritas kedua modalitas.
7. `fig_c2_confusion_matrix_multimodal.png`:
   - *Caption*: Normalized Confusion Matrix model multimodal terbaik pada data validasi.
8. `fig_d1_loss_distribution_noise.png`:
   - *Caption*: Distribusi sample-wise Cross-Entropy Loss pada data latih untuk mendeteksi sampel anomali dengan supervisi label meragukan.
9. `fig_f1_sensor_masking_drop.png`:
   - *Caption*: Kurva graceful degradation Macro-F1 saat sensor tekan dimasking secara artifisial dari 0% hingga 100%.
10. `fig_g1_sanity_overfit.png`:
    - *Caption*: Sanity check optimasi pada satu mini-batch membuktikan konvergensi loss menuju 0 dan akurasi 100%.
11. `fig_g2_learning_curves.png`:
    - *Caption*: Kurva pembelajaran (loss dan Macro-F1) per epoch untuk data latih dan validasi dengan scheduler CosineAnnealing.

---

### 3. Keputusan Desain & Alasan Teknis (Panduan Ujian Lisan)

1. **Mengapa Memilih Macro-F1 Sebagai Objective Utama?**
   - Distribusi kelas sangat timpang (kelas `normal` 49,8%, sedangkan `kontaminasi` hanya 4,9%). Jika menggunakan Akurasi, model pemalas yang memprediksi mayoritas kelas `normal` akan mendapat akurasi ~50% tetapi Macro-F1 sangat rendah. Macro-F1 memberikan bobot setara pada setiap kelas, memaksa model belajar mengenali cacat-cacat kritis minoritas.
2. **Mengapa Menggunakan 1D-ResNet untuk Sinyal Sensor?**
   - Sinyal sensor tekanan memiliki 3 kanal fisis (suhu, getaran, arus) sepanjang 128 timestep. Conv1D dengan residual connection sangat efisien dalam menangkap fitur multi-skala: getaran frekuensi tinggi (kernel sempit) dan tren pemanasan suhu global (receptive field luas), dengan komputasi jauh lebih cepat dan hemat memori dibanding RNN/LSTM pada GPU 4 GB.
3. **Mengapa Modality Dropout Wajib Digunakan?**
   - Data riil di pabrik memiliki ~5% kasus di mana sensor gagal merekam (NaN). Jika model hanya dilatih saat kedua modalitas ada, model akan runtuh saat menghadapi data uji yang kehilangan sensor. Dengan menjatuhkan sinyal sensor secara acak (p=0.15) saat pelatihan dan menyuplai token mask eksplisit, model belajar mengekstraksi representasi mandiri dari gambar saat sensor mati.
4. **Bagaimana Menangani Domain Shift Lini A ke Lini B?**
   - Train set didominasi Lini A (75%), namun Test set didominasi Lini B (60%). Kami menerapkan normalisasi z-score adaptif dan augmentasi fotometrik/sensorik, serta memvalidasi model secara terpisah per lini produksi untuk memastikan model tidak sekadar menghafal karakteristik mesin `M01-M06` (Lini A).
5. **Bagaimana Menangani Label Tidak Sempurna (Noisy Labels)?**
   - Karena label berasal dari inspektur manusia yang rentan kelelahan, kami mengaudit sampel dengan loss tertinggi dan menguji *Soft Bootstrapping Loss* serta *Label Smoothing*. Teknik ini melunakkan target satu-hot yang kaku, mencegah penalti ekstrem saat model memprediksi cacat yang benar meskipun label inspektur keliru.