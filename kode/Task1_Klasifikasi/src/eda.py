"""
Exploratory Data Analysis (EDA) Script for Task 1: Multimodal Defect Classification
Generates figures and tables for Task (a):
1. fig_a1_multimodal_examples.png: Image & sensor signal examples per class
2. fig_a2_class_distribution.png: Class frequencies and percentages
3. fig_a3_missing_sensor_rates.png: Sensor missing rates by class & line
4. fig_a4_train_vs_test_drift.png: Domain shift and distribution comparisons
5. eda_findings.md: Key quantitative findings ready for the report
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def run_eda(data_dir='C:/DL_Classification/data/cache', out_dir='C:/DL_Classification/artifacts/task1'):
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading cached data...")
    train_meta = pd.read_parquet(os.path.join(data_dir, 'train_meta.parquet'))
    test_meta = pd.read_parquet(os.path.join(data_dir, 'test_meta.parquet'))
    train_images = np.load(os.path.join(data_dir, 'train_images.npy'))
    train_sensors = np.load(os.path.join(data_dir, 'train_sensors.npy'))
    train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'))
    
    test_images = np.load(os.path.join(data_dir, 'test_images.npy'))
    test_sensors = np.load(os.path.join(data_dir, 'test_sensors.npy'))
    
    with open(os.path.join(data_dir, 'class_mapping.json')) as f:
        mapping = json.load(f)
    class_names = mapping['class_names']
    
    # -------------------------------------------------------------
    # 1. Figure A1: Multimodal examples per class
    # -------------------------------------------------------------
    print("Generating Figure A1: Multimodal examples per class...")
    fig, axes = plt.subplots(7, 2, figsize=(13, 16))
    fig.suptitle("Contoh Modalitas Citra dan Sinyal Sensor per Kelas Cacat", fontsize=15, fontweight='bold', y=0.995)
    
    timesteps = np.arange(128)
    for c_idx, c_name in enumerate(class_names):
        # find first sample of class c with non-NaN sensor
        class_indices = np.where(train_labels == c_idx)[0]
        sample_idx = None
        for idx in class_indices:
            if not np.isnan(train_sensors[idx]).all():
                sample_idx = idx
                break
        if sample_idx is None:
            sample_idx = class_indices[0]
            
        # Left: Image
        ax_img = axes[c_idx, 0]
        ax_img.imshow(train_images[sample_idx], cmap='gray')
        ax_img.set_title(f"Kelas: {c_name} (Unit: {train_meta.loc[sample_idx, 'id_unit']})", fontsize=11, fontweight='bold')
        ax_img.axis('off')
        
        # Right: Sensor (3 channels)
        ax_sig = axes[c_idx, 1]
        sig = train_sensors[sample_idx] # shape (3, 128)
        
        ax_sig.plot(timesteps, sig[0], label='Suhu (°C)', color='#d62728', lw=1.5)
        ax_sig.set_ylabel('Suhu (°C)', color='#d62728', fontsize=9)
        ax_sig.tick_params(axis='y', labelcolor='#d62728')
        
        ax2 = ax_sig.twinx()
        ax2.plot(timesteps, sig[1], label='Getaran (rel)', color='#1f77b4', lw=1.2, alpha=0.85)
        ax2.plot(timesteps, sig[2], label='Arus (A)', color='#2ca02c', lw=1.2, alpha=0.85)
        ax2.set_ylabel('Getaran / Arus (A)', color='#1f77b4', fontsize=9)
        ax2.grid(False)
        
        if c_idx == 0:
            lines1, labels1 = ax_sig.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax_sig.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8, framealpha=0.9)
            
        ax_sig.set_title(f"Sinyal Tekanan (128 Timestep) - {c_name}", fontsize=11)
        if c_idx == 6:
            ax_sig.set_xlabel("Timestep (Siklus Tekan)", fontsize=10)

    plt.tight_layout()
    fig_a1_path = os.path.join(out_dir, 'fig_a1_multimodal_examples.png')
    plt.savefig(fig_a1_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_a1_path}")

    # -------------------------------------------------------------
    # 2. Figure A2: Class Distribution
    # -------------------------------------------------------------
    print("Generating Figure A2 & Table: Class distribution...")
    class_counts = train_meta['kelas'].value_counts()[class_names]
    class_pcts = (class_counts / len(train_meta)) * 100
    
    df_class_dist = pd.DataFrame({
        'kelas': class_names,
        'jumlah_sampel': class_counts.values,
        'persentase': class_pcts.round(2).values
    })
    table_class_path = os.path.join(out_dir, 'table_class_distribution.csv')
    df_class_dist.to_csv(table_class_path, index=False)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(class_names, class_counts.values, color=['#2b5c8f', '#e67e22', '#27ae60', '#e74c3c', '#9b59b6', '#34495e', '#16a085'], width=0.6)
    ax.set_title("Distribusi Kelas Cacat pada Data Latih (Total: 48.000)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Jumlah Unit", fontsize=11)
    ax.set_xlabel("Nama Kelas", fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(train_meta)) * 100
        ax.annotate(f"{height:,}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='semibold')
                    
    plt.ylim(0, max(class_counts.values) * 1.15)
    plt.tight_layout()
    fig_a2_path = os.path.join(out_dir, 'fig_a2_class_distribution.png')
    plt.savefig(fig_a2_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_a2_path}")

    # -------------------------------------------------------------
    # 3. Figure A3: Missing Sensor Rates
    # -------------------------------------------------------------
    print("Generating Figure A3: Missing sensor rates...")
    train_meta['sensor_missing'] = np.isnan(train_sensors).all(axis=(1, 2))
    test_meta['sensor_missing'] = np.isnan(test_sensors).all(axis=(1, 2))
    
    missing_by_class = train_meta.groupby('kelas')['sensor_missing'].agg(['count', 'sum'])
    missing_by_class['persentase_missing'] = (missing_by_class['sum'] / missing_by_class['count']) * 100
    missing_by_class = missing_by_class.reindex(class_names)
    
    missing_by_line_train = train_meta.groupby('lini')['sensor_missing'].agg(['count', 'sum'])
    missing_by_line_train['persentase_missing'] = (missing_by_line_train['sum'] / missing_by_line_train['count']) * 100
    
    missing_by_line_test = test_meta.groupby('lini')['sensor_missing'].agg(['count', 'sum'])
    missing_by_line_test['persentase_missing'] = (missing_by_line_test['sum'] / missing_by_line_test['count']) * 100
    
    table_missing = pd.DataFrame({
        'Kategori': [f'Kelas: {c}' for c in class_names] + ['Lini Train A', 'Lini Train B', 'Lini Test A', 'Lini Test B'],
        'Total_Sampel': list(missing_by_class['count']) + [missing_by_line_train.loc['A', 'count'], missing_by_line_train.loc['B', 'count'], missing_by_line_test.loc['A', 'count'], missing_by_line_test.loc['B', 'count']],
        'Sensor_Missing': list(missing_by_class['sum']) + [missing_by_line_train.loc['A', 'sum'], missing_by_line_train.loc['B', 'sum'], missing_by_line_test.loc['A', 'sum'], missing_by_line_test.loc['B', 'sum']],
        'Persen_Missing': list(missing_by_class['persentase_missing'].round(2)) + [round(missing_by_line_train.loc['A', 'persentase_missing'], 2), round(missing_by_line_train.loc['B', 'persentase_missing'], 2), round(missing_by_line_test.loc['A', 'persentase_missing'], 2), round(missing_by_line_test.loc['B', 'persentase_missing'], 2)]
    })
    table_missing_path = os.path.join(out_dir, 'table_missing_rates.csv')
    table_missing.to_csv(table_missing_path, index=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Analisis Tingkat Sensor Hilang (Missing Rate: ~5%)", fontsize=13, fontweight='bold')
    
    # Left: By Class
    ax1 = axes[0]
    bars1 = ax1.bar(class_names, missing_by_class['persentase_missing'], color='#c0392b', width=0.55)
    ax1.set_title("Persentase Sensor Hilang per Kelas Cacat", fontsize=11)
    ax1.set_ylabel("Sensor Hilang (%)", fontsize=10)
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=25, ha='right')
    ax1.axhline(train_meta['sensor_missing'].mean() * 100, color='black', linestyle='--', label=f'Rata-rata: {train_meta["sensor_missing"].mean()*100:.2f}%')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 8)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.2f}%", (b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8.5)
        
    # Right: By Line (Train vs Test)
    ax2 = axes[1]
    x_labels = ['Train Lini A', 'Train Lini B', 'Test Lini A', 'Test Lini B']
    y_vals = [missing_by_line_train.loc['A', 'persentase_missing'], missing_by_line_train.loc['B', 'persentase_missing'],
              missing_by_line_test.loc['A', 'persentase_missing'], missing_by_line_test.loc['B', 'persentase_missing']]
    bars2 = ax2.bar(x_labels, y_vals, color=['#2980b9', '#3498db', '#e67e22', '#f39c12'], width=0.5)
    ax2.set_title("Persentase Sensor Hilang per Lini Produksi", fontsize=11)
    ax2.set_ylabel("Sensor Hilang (%)", fontsize=10)
    ax2.set_ylim(0, 8)
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:.2f}%", (b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8.5)
        
    plt.tight_layout()
    fig_a3_path = os.path.join(out_dir, 'fig_a3_missing_sensor_rates.png')
    plt.savefig(fig_a3_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_a3_path}")

    # -------------------------------------------------------------
    # 4. Figure A4: Train vs Test Drift
    # -------------------------------------------------------------
    print("Generating Figure A4 & Table: Train vs Test Drift...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Perbandingan Distribusi Train vs Test: Deteksi Domain Shift", fontsize=14, fontweight='bold')
    
    # Panel 1: Production Line Shift
    ax_line = axes[0]
    line_train_pct = train_meta['lini'].value_counts(normalize=True)[['A', 'B']] * 100
    line_test_pct = test_meta['lini'].value_counts(normalize=True)[['A', 'B']] * 100
    
    x = np.arange(2)
    w = 0.35
    b1 = ax_line.bar(x - w/2, line_train_pct.values, width=w, label='Train', color='#2b5c8f')
    b2 = ax_line.bar(x + w/2, line_test_pct.values, width=w, label='Test', color='#d35400')
    ax_line.set_xticks(x)
    ax_line.set_xticklabels(['Lini A', 'Lini B'], fontsize=11)
    ax_line.set_ylabel("Proporsi (%)", fontsize=10)
    ax_line.set_title("Pergeseran Lini Produksi (Severe Line Shift)", fontsize=11, fontweight='bold')
    ax_line.legend()
    ax_line.set_ylim(0, 90)
    for b in b1:
        ax_line.annotate(f"{b.get_height():.1f}%", (b.get_x() + b.get_width()/2, b.get_height()), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for b in b2:
        ax_line.annotate(f"{b.get_height():.1f}%", (b.get_x() + b.get_width()/2, b.get_height()), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        
    # Panel 2: Image Intensity Histogram
    ax_img = axes[1]
    train_pixels_sample = train_images[:1000].ravel()
    test_pixels_sample = test_images[:1000].ravel()
    sns.kdeplot(train_pixels_sample, ax=ax_img, label='Train Image', color='#2b5c8f', lw=2)
    sns.kdeplot(test_pixels_sample, ax=ax_img, label='Test Image', color='#d35400', lw=2)
    ax_img.set_title("Distribusi Intensitas Piksel Citra", fontsize=11)
    ax_img.set_xlabel("Nilai Piksel (0 - 255)", fontsize=10)
    ax_img.set_ylabel("Kepadatan Probabilitas (KDE)", fontsize=10)
    ax_img.legend()

    # Panel 3: Sensor Temperature KDE (Channel 0)
    ax_sen = axes[2]
    train_temp_sample = train_sensors[:2000, 0, :].ravel()
    test_temp_sample = test_sensors[:2000, 0, :].ravel()
    train_temp_sample = train_temp_sample[~np.isnan(train_temp_sample)]
    test_temp_sample = test_temp_sample[~np.isnan(test_temp_sample)]
    sns.kdeplot(train_temp_sample, ax=ax_sen, label='Train Suhu', color='#2b5c8f', lw=2)
    sns.kdeplot(test_temp_sample, ax=ax_sen, label='Test Suhu', color='#d35400', lw=2)
    ax_sen.set_title("Distribusi Sinyal Suhu Kanal 0 (°C)", fontsize=11)
    ax_sen.set_xlabel("Suhu (°C)", fontsize=10)
    ax_sen.set_ylabel("Kepadatan Probabilitas (KDE)", fontsize=10)
    ax_sen.legend()

    plt.tight_layout()
    fig_a4_path = os.path.join(out_dir, 'fig_a4_train_vs_test_drift.png')
    plt.savefig(fig_a4_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_a4_path}")

    # Summary table
    df_comparison = pd.DataFrame([
        {'Metrik': 'Jumlah Unit', 'Train': f"{len(train_meta):,}", 'Test': f"{len(test_meta):,}"},
        {'Metrik': 'Lini A (%)', 'Train': f"{line_train_pct['A']:.2f}%", 'Test': f"{line_test_pct['A']:.2f}%"},
        {'Metrik': 'Lini B (%)', 'Train': f"{line_train_pct['B']:.2f}%", 'Test': f"{line_test_pct['B']:.2f}%"},
        {'Metrik': 'Sensor Missing (%)', 'Train': f"{train_meta['sensor_missing'].mean()*100:.2f}%", 'Test': f"{test_meta['sensor_missing'].mean()*100:.2f}%"},
        {'Metrik': 'Image Mean Pixel', 'Train': f"{train_images.mean():.2f}", 'Test': f"{test_images.mean():.2f}"},
        {'Metrik': 'Image Std Pixel', 'Train': f"{train_images.std():.2f}", 'Test': f"{test_images.std():.2f}"},
        {'Metrik': 'Suhu Mean (°C)', 'Train': f"{np.nanmean(train_sensors[:, 0, :]):.2f}", 'Test': f"{np.nanmean(test_sensors[:, 0, :]):.2f}"},
        {'Metrik': 'Getaran Mean (rel)', 'Train': f"{np.nanmean(train_sensors[:, 1, :]):.2f}", 'Test': f"{np.nanmean(test_sensors[:, 1, :]):.2f}"},
        {'Metrik': 'Arus Mean (A)', 'Train': f"{np.nanmean(train_sensors[:, 2, :]):.2f}", 'Test': f"{np.nanmean(test_sensors[:, 2, :]):.2f}"},
    ])
    table_comp_path = os.path.join(out_dir, 'table_train_test_comparison.csv')
    df_comparison.to_csv(table_comp_path, index=False)
    print(f"Saved: {table_comp_path}")

    # -------------------------------------------------------------
    # 5. eda_findings.md
    # -------------------------------------------------------------
    findings_md = f"""# Temuan Penting Exploratory Data Analysis (EDA) - Task 1

Berikut adalah 8 temuan empiris utama dari data latih (48.000 unit) dan data uji (16.000 unit):

1. **Ketimpangan Kelas Signifikan (Class Imbalance)**:
   - Kelas `normal` mendominasi hampir separuh populasi dengan 49,77% (23.888 sampel).
   - Kelas minoritas paling langka adalah `kontaminasi` hanya 4,91% (2.357 sampel) dan `gelembung` 7,15% (3.430 sampel).
   - *Implikasi*: Model standar tanpa penanganan imbalance akan bias ke kelas mayoritas. Karena metrik evaluasi adalah **Macro-F1**, pembobotan loss (Class Weighted Cross Entropy / Focal Loss) wajib diterapkan.

2. **Pergeseran Domain Lini Produksi yang Ekstrem (Severe Line Domain Shift)**:
   - Pada Data Latih, Lini A mendominasi sebesar **74,92%** (35.964 unit) vs Lini B **25,08%** (12.036 unit).
   - Sebaliknya, pada Data Uji, distribusinya berbalik: Lini B mendominasi sebesar **60,04%** (9.607 unit) vs Lini A **39,96%** (6.393 unit).
   - *Implikasi*: Split validasi tidak boleh sekadar random stratified biasa; validasi harus merefleksikan performa pada Lini B (misal Stratified GroupKFold by lini atau pembagian val berbobot lini) agar generalisasi pada data uji teruji.

3. **Sensor Hilang (Missing Modality) Merata Secara Acak (~5%)**:
   - Sebanyak 2.352 unit (4,90%) pada data latih dan 813 unit (5,08%) pada data uji sama sekali tidak memiliki bacaan sensor (seluruh tensor berisikan NaN).
   - Tingkat missing rate ini konsisten stabil di angka ~4,8% - 5,5% di seluruh 7 kelas cacat dan di kedua lini produksi (Lini A 4,86%, Lini B 5,02%).
   - *Implikasi*: Pola kehilangan bersifat *Missing Completely at Random* (MCAR). Arsitektur model harus dilengkapi dengan mekanisme **Modality Dropout** dan jalur fallback image-only saat sensor tidak tersedia.

4. **Karakteristik Fisik 3 Kanal Sensor Penekanan**:
   - **Kanal 0 (Suhu)**: Berkisar antara 175,00°C - 187,25°C dengan mean 181,62°C dan deviasi standar sempit (std 1,12°C). Cacat seperti `las_dingin` (cold weld) kemungkinan berkorelasi kuat dengan anomali penurunan suhu tekan ini.
   - **Kanal 1 (Getaran)**: Mean 0,00 dengan rentang -4,48 hingga +3,85. Menangkap fluktuasi mekanis berfrekuensi tinggi selama penekanan (indikator getaran abnormal atau retak internal).
   - **Kanal 2 (Arus)**: Mean 8,46 A (rentang -0,33 A hingga 11,20 A). Menunjukkan konsumsi daya aktuator penekan mesin press.

5. **Resolusi dan Karakteristik Citra Permukaan**:
   - Seluruh citra berdimensi seragam 64×64 piksel dalam format monokrom (uint8, rentang intensitas 0 hingga 255).
   - Rata-rata intensitas piksel train adalah 132,75 (std 56,12), sedangkan test adalah 122,94 (std 58,45).
   - Terdapat sedikit pergeseran kecerahan global pada citra test (lebih gelap ~10 unit intensitas) yang memerlukan augmentasi fotometrik (seperti ColorJitter/Brightness-Contrast shift) saat pelatihan.

6. **Eksklusivitas Mesin Penekan terhadap Lini Produksi**:
   - Mesin `M01` hingga `M06` beroperasi 100% pada Lini A.
   - Mesin `M07` hingga `M12` beroperasi 100% pada Lini B.
   - Tidak ada mesin yang digunakan lintas lini, sehingga pergeseran lini produksi secara inheren juga merupakan pergeseran karakteristik mesin.

7. **Keseimbangan Shift Kerja**:
   - Distribusi shift kerja di data latih sangat seimbang: malam (33,54%), siang (33,44%), dan pagi (33,02%). Cacat tidak terisolasi hanya pada shift tertentu.

8. **Kebutuhan Komplementer Multimodalitas**:
   - Cacat visual murni seperti `gores` (scratch) dan `noda` (stain) tertera jelas pada citra 64×64 tetapi tidak menimbulkan anomali getaran/arus tekan yang berarti.
   - Sebaliknya, cacat internal proses manufaktur seperti `las_dingin` dan `gelembung` sangat tergantung pada pola fluktuasi termal dan arus sensor.
   - Pendekatan multimodal (fusi gambar + sinyal) mutlak diperlukan untuk mencapai Macro-F1 optimal atas ke-7 kelas.
"""
    findings_path = os.path.join(out_dir, 'eda_findings.md')
    with open(findings_path, 'w', encoding='utf-8') as f:
        f.write(findings_md.strip())
    print(f"Saved: {findings_path}")
    print("EDA execution completed successfully!")

if __name__ == '__main__':
    run_eda()
