# Temuan Penting Exploratory Data Analysis (EDA) - Task 1

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