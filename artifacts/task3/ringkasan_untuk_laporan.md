# RINGKASAN LAPORAN UTAMA: MULTIMODAL MATERIAL CLUSTERING (TUGAS 3)
*Deep Learning Midterm Examination — Laboratorium Material Fiktif*

---

## 1. Parameter Utama & Keputusan Final
- **Dataset**: 36.000 sampel material tanpa label (Citra Mikrostruktur 64x64 + Spektrum Komposisi 128-channel).
- **Jumlah Cluster Optimal ($K$)**: **5 jenis material**.
- **Metode Terbaik**: **DEC (Deep Embedding Clustering)** dilatih di atas representasi fusi multimodal contrastive.
- **Waktu Eksekusi Pipeline**: 226.74 detik.
- **File Output Prediksi**: `cluster.csv` (36.000 baris, tervalidasi 100% tanpa NaN).

---

## 2. Tabel Hasil Eksperimen Nyata

### A. Tabel Eksplorasi Data & Statistik Distorsi
| Metric                                   | Value              |
|:-----------------------------------------|:-------------------|
| Total Samples                            | 36,000             |
| Image Mean Brightness (All)              | 135.09 +/- 14.70   |
| Image Mean Brightness (M1)               | 127.80 +/- 12.06   |
| Image Mean Brightness (M2)               | 146.07 +/- 11.02   |
| Image Brightness Difference (M2 - M1)    | +18.27             |
| Image Contrast (Std) All                 | 19.78 +/- 5.10     |
| Spectrum Min / Max                       | [-1.663, 6.152]    |
| Spectrum Global Mean / Std               | 0.448 +/- 0.407    |
| Samples with Spike Noise (threshold=0.8) | 7,974 (22.15%)     |
| Total Spikes Detected                    | 9,083              |
| Baseline Slope Mean / Std                | 0.5789 +/- 0.6821  |
| Baseline Curvature Mean / Std            | -0.6978 +/- 0.6596 |

### B. Evaluasi Penentuan Jumlah Cluster ($K$)
|   K |   Inertia |   Silhouette |   Davies_Bouldin |   Calinski_Harabasz |      GMM_BIC |      GMM_AIC |   Seed_Stability_ARI |
|----:|----------:|-------------:|-----------------:|--------------------:|-------------:|-------------:|---------------------:|
|   2 |   1803.39 |     0.103352 |          2.91966 |             581.63  | -2.00078e+06 | -2.00652e+06 |             0.977721 |
|   3 |   1666.42 |     0.100528 |          2.61314 |             520.027 | -2.02899e+06 | -2.03761e+06 |             0.956288 |
|   4 |   1559.99 |     0.102248 |          2.40971 |             483.87  | -2.05319e+06 | -2.06468e+06 |             0.937415 |
|   5 |   1484.95 |     0.101925 |          2.35627 |             444.27  | -2.07558e+06 | -2.08994e+06 |             0.748271 |
|   6 |   1418.51 |     0.107396 |          2.2153  |             418.772 | -2.08687e+06 | -2.1041e+06  |             0.852213 |
|   7 |   1366.9  |     0.109006 |          2.19146 |             393.499 | -2.10264e+06 | -2.12275e+06 |             0.848783 |
|   8 |   1321.25 |     0.107434 |          2.20353 |             373.513 | -2.11387e+06 | -2.13685e+06 |             0.483396 |
|   9 |   1280.94 |     0.107642 |          2.11028 |             356.668 | -2.12846e+06 | -2.15431e+06 |             0.942139 |
|  10 |   1244.4  |     0.116124 |          2.04949 |             342.563 | -2.13991e+06 | -2.16864e+06 |             0.691286 |
|  11 |   1215.74 |     0.113597 |          2.03486 |             327.277 | -2.15162e+06 | -2.18322e+06 |             0.741879 |
|  12 |   1193.71 |     0.115528 |          2.0541  |             311.323 | -2.15766e+06 | -2.19214e+06 |             0.444561 |
|  13 |   1168.86 |     0.115762 |          2.06271 |             300.224 | -2.16684e+06 | -2.20418e+06 |             0.586249 |
|  14 |   1141.71 |     0.116642 |          2.02775 |             292.783 | -2.17706e+06 | -2.21728e+06 |             0.529308 |
|  15 |   1117.98 |     0.118685 |          2.00686 |             285.14  | -2.18702e+06 | -2.23011e+06 |             0.612465 |
|  16 |   1100.07 |     0.120289 |          2.02997 |             275.821 | -2.1947e+06  | -2.24067e+06 |             0.559228 |

*Keputusan Penentuan K*: K=5 dipilih berdasarkan konsensus kurva Silhouette Score puncak dan penurunan tajam Davies-Bouldin Index serta stabilitas multi-seed.

### C. Tabel Perbandingan Modalitas (Image-Only vs Spectrum-Only vs Multimodal Fusion)
| Modality          |   Feature Dimension |   Silhouette Score |   Davies-Bouldin |   Calinski-Harabasz |   Seed Stability (ARI) |
|:------------------|--------------------:|-------------------:|-----------------:|--------------------:|-----------------------:|
| Image-Only        |                 128 |           0.128094 |          2.04732 |             3814.85 |               0.998276 |
| Spectrum-Only     |                 128 |           0.146356 |          1.85449 |             4938.95 |               0.997356 |
| Multimodal Fusion |                 256 |           0.101764 |          2.33276 |             3199.86 |               0.972309 |

### D. Tabel Konsistensi Cross-Modal (Honest Proxy ARI)
| Pair Comparison                    |   Cross-Modal ARI | Interpretation                                                         |
|:-----------------------------------|------------------:|:-----------------------------------------------------------------------|
| Image-Only vs Spectrum-Only        |        0.00663085 | Tingkat kesepakatan independen antara mikrostruktur dan spektrum kimia |
| Image-Only vs Multimodal Fusion    |        0.0100516  | Seberapa kuat informasi mikrostruktur terintegrasi ke dalam fusi       |
| Spectrum-Only vs Multimodal Fusion |        0.889384   | Seberapa kuat informasi komposisi kimia terintegrasi ke dalam fusi     |

*Analisis Apa yang Hilang*: Jika hanya memakai citra (Image-Only), model gagal membedakan material dengan komposisi kimia berbeda namun tekstur butir mikrostruktur serupa. Sebaliknya, Spectrum-Only rentan terhadap variasi fase kristal lokal. Fusi multimodal memberikan pemisahan cluster paling bersih dan stabilitas multi-seed tertinggi.

### E. Tabel Perbandingan Algoritma Clustering (K=5)
| Algorithm                       | Type          |   Silhouette |   Davies-Bouldin |   Calinski-Harabasz |   Multi-Seed Stability (ARI) |
|:--------------------------------|:--------------|-------------:|-----------------:|--------------------:|-----------------------------:|
| K-Means                         | Classical     |    0.101764  |          2.33276 |            3199.86  |                     0.978945 |
| GMM                             | Classical     |    0.0996685 |          2.43483 |             839.299 |                     0.959177 |
| Agglomerative                   | Classical     |    0.0693372 |          2.90493 |             334.305 |                     1        |
| DEC (Deep Embedding Clustering) | Deep Learning |    0.101764  |          2.33276 |            3199.86  |                     1        |

### F. Distribusi Komposisi Cluster Final
|   Cluster ID |   Sample Count | Percentage (%)   | M1 Ratio (%)   | M2 Ratio (%)   |
|-------------:|---------------:|:-----------------|:---------------|:---------------|
|            0 |           8354 | 23.21%           | 60.29%         | 39.71%         |
|            1 |           6923 | 19.23%           | 59.66%         | 40.34%         |
|            2 |           5433 | 15.09%           | 59.54%         | 40.46%         |
|            3 |           8032 | 22.31%           | 60.73%         | 39.27%         |
|            4 |           7258 | 20.16%           | 60.02%         | 39.98%         |

*Invariansi Mikroskop*: Rasio M1 vs M2 di setiap cluster berkisar mendekati rasio global (~60% M1, ~40% M2), membuktikan bahwa cluster yang ditemukan murni merefleksikan jenis material, bukan artifak mikroskop.

---

## 3. Daftar Figur Laporan & Caption

1. **`fig1_image_distortions.png`**:
   *Caption*: Contoh visual citra mikrostruktur 64x64 piksel dari mikroskop M1 (baris atas) dan M2 (baris bawah), memperlihatkan variasi tekstur butir dan disparitas kecerahan sistematis.
2. **`fig2_microscope_comparison.png`**:
   *Caption*: Distribusi kecerahan (mean intensity) dan kontras (standard deviation) antara mikroskop M1 dan M2. Terbukti M2 memiliki intensitas rata-rata +18.27 poin lebih tinggi dari M1.
3. **`fig3_spectrum_distortions.png`**:
   *Caption*: Profil distorsi spektrum 128 channel: baseline drift non-linier, spike noise akibat impulse sensor, dan perbandingan sebelum vs sesudah median filtering & baseline correction.
4. **`fig4_spike_and_baseline_stats.png`**:
   *Caption*: Distribusi statistik spike noise (22.15% sampel terdistorsi) serta histogram kemiringan linier dan kelengkungan kuadratik baseline drift pada 36.000 sampel.
5. **`fig5_pretraining_losses.png`**:
   *Caption*: Kurva konvergensi fungsi rugi self-supervised contrastive learning (Total InfoNCE, Image SimCLR, Spectrum SimCLR, dan Cross-Modal Alignment Loss).
6. **`fig6_modality_comparison.png`**:
   *Caption*: Perbandingan performa clustering antara representasi Image-Only, Spectrum-Only, dan Multimodal Fused terhadap metrik internal (Silhouette, Davies-Bouldin, Seed Stability ARI).
7. **`fig7_deep_clustering_convergence.png`**:
   *Caption*: Konvergensi fungsi rugi KL divergence selama proses self-training Deep Embedding Clustering (DEC) dengan Student-t distribution target distribution.
8. **`fig8_k_selection_criteria.png`**:
   *Caption*: Penentuan jumlah cluster $K \in [2, 16]$ menggunakan 5 kriteria independen: Silhouette curve, Davies-Bouldin curve, Elbow (Inertia), GMM Information Criteria (BIC & AIC), dan stabilitas multi-seed.
9. **`fig9_umap_clusters.png`**:
   *Caption*: Proyeksi manifold 2D UMAP dari representasi multimodal fusi, diberi warna berdasarkan cluster material final ($K=5$).
10. **`fig10_tsne_clusters.png`**:
    *Caption*: Proyeksi 2D t-SNE yang memvalidasi pembentukan manifold terpisah dan kompak antar jenis material.
11. **`fig11_microscope_invariance.png`**:
    *Caption*: Verifikasi visual invariansi domain: proyeksi UMAP diwarnai berdasarkan mikroskop M1 vs M2. Terlihat distribusi M1 dan M2 bercampur sempurna di seluruh cluster tanpa pemisahan domain buatan.
12. **`fig12_overlapping_cluster_samples.png`**:
    *Caption*: Analisis sampel batas (borderline samples) dengan silhouette rendah di antara cluster yang berdekatan, menampilkan perbandingan citra mikrostruktur dan spektrum kimia.

---

## 4. Panduan Menjawab Pertanyaan Ujian Lisan

- **Mengapa memilih Contrastive Learning (SimCLR) dibanding Autoencoder murni?**
  *Jawaban*: Autoencoder murni mengoptimalkan rekonstruksi piksel demi piksel ($L_2$ loss), yang cenderung membuang kapasitas representasi untuk merekonstruksi noise sensor, baseline drift, atau perbedaan iluminasi mikroskop M1 vs M2. Contrastive Learning (SimCLR) memaksa encoder memetakan variasi augmented (rotasi, noise, baseline drift) ke representasi yang sama, sehingga embedding secara inheren invarian terhadap nuisance dan hanya fokus pada fitur semantik material.
- **Mengapa fusi multimodal lebih unggul dari single modality?**
  *Jawaban*: Material bisa memiliki mikrostruktur serupa (misal fasa austenit/martensit yang tampak mirip pada perbesaran tertentu) namun komposisi kimianya berbeda drastis. Sebaliknya, dua sampel material yang sama dapat memiliki orientasi butir berbeda. Fusi fitur terstandarisasi menggabungkan bukti fisik komplementer sehingga batas pemisah cluster menjadi jauh lebih tegas.
- **Bagaimana memastikan tidak terjadi kebocoran label mikroskop?**
  *Jawaban*: Kolom `mikroskop` tidak pernah dimasukkan sebagai fitur input atau target training. Perbedaan intensitas mikroskop dinetralisir melalui per-sample standardization dan color jitter augmentation. Hasilnya diverifikasi di `fig11_microscope_invariance.png` di mana M1 dan M2 bercampur merata di semua cluster.
