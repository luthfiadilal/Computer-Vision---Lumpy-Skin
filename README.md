# 🐄 Lumpy Skin Disease Detection using Deep Learning (InceptionV3 / GoogLeNet)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Computer Vision](https://img.shields.io/badge/Task-Computer%20Vision%20%7C%20Classification-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

Repositori ini berisi proyek Computer Vision berbasis Deep Learning untuk mendeteksi penyakit **Lumpy Skin Disease (LSD)** pada kulit sapi berdasarkan citra digital. Model dibangun menggunakan pendekatan *Transfer Learning* dan *Fine-Tuning* dengan arsitektur dasar **InceptionV3** (bagian dari famili arsitektur GoogLeNet/Inception).

---

## 📌 Latar Belakang & Ringkasan Proyek

**Lumpy Skin Disease (LSD)** adalah penyakit infeksius pada sapi yang disebabkan oleh virus dari genus *Capripoxvirus*. Gejala utama ditandai dengan munculnya nodul atau benjolan khas pada kulit sapi di seluruh tubuh. Penyakit ini dapat menyebabkan penurunan produksi susu, kerusakan kulit, gangguan reproduksi, hingga kerugian ekonomi yang signifikan bagi peternak.

Deteksi dini secara manual sering kali membutuhkan keahlian dokter hewan dan memakan waktu. Proyek ini bertujuan untuk membangun model klasifikasi citra otomatis yang mampu membedakan sapi yang terinfeksi (*Lumpy Skin*) dan sapi sehat (*Normal Skin*) secara cepat dan akurat.

---

## ⚙️ Fitur Utama

- **Klasifikasi Biner Citra**: Mengklasifikasikan citra kulit sapi ke dalam 2 kelas:
  - `Lumpy Skin`: Kulit sapi yang terjangkit nodul LSD.
  - `Normal Skin`: Kulit sapi sehat/normal.
- **Transfer Learning & Fine-Tuning**: Menggunakan pretrained weights ImageNet dari arsitektur InceptionV3 dengan fine-tuning 30 layer teratas untuk ekstraksi fitur spesifik nodul kulit.
- **Augmentasi Citra Komprehensif**: Penyesuaian rotasi, zoom, geser (shift), shear, pembalikan horizontal/vertikal, kecerahan, dan pergeseran kanal warna untuk memperkuat generalisasi model.
- **Eksplorasi Preprocessing Citra**: Eksperimen segmentasi ambang batas (*thresholding*) berbasis OpenCV untuk menonjolkan fitur tekstur kulit.
- **Handling Imbalance & Scheduler**: Pemanfaatan *class weight balancing* dan *exponential learning rate decay* untuk stabilitas konvergensi saat training.
- **Visualisasi & Evaluasi**: Pelacakan metrik accuracy/loss curve dan matriks konfusi (*confusion matrix*).

---

## 🧠 Arsitektur Model

Arsitektur model dirancang dengan skema *Transfer Learning*:
1. **Base Model**: `InceptionV3` (Pretrained di ImageNet, input shape `224x224x3`).
   - Pembekuan layer (*freezing*) awal dan membuka 30 layer teratas untuk fine-tuning.
2. **Custom Classification Head**:
   - `GlobalAveragePooling2D`
   - `BatchNormalization`
   - `Dense(256, activation='relu')`
   - `Dropout(0.5)`
   - `BatchNormalization`
   - `Dense(128, activation='relu')`
   - `Dropout(0.3)`
   - `Dense(2, activation='softmax')`
3. **Optimizer & Loss**:
   - Optimizer: `Adam` dengan `ExponentialDecay` learning rate schedule ($lr_{init} = 10^{-4}$).
   - Loss: `Categorical Crossentropy`.

---

## 📊 Dataset & Distribusi Data

Dataset dibagi menjadi tiga set utama:

| Kelas | Training (70%) | Testing (20%) | Validation (10%) | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Lumpy Skin** | ~471 | ~135 | ~68 | ~674 |
| **Normal Skin** | ~477 | ~136 | ~69 | ~682 |
| **Total** | **~948** | **~271** | **~137** | **~1356** |

---

## 📈 Hasil & Evaluasi Sementara

Berdasarkan pengujian pada data uji (*test set*):
- **Akurasi Pengujian (Test Accuracy)**: $\approx 84\% - 87\%$
- **Loss Pengujian (Test Loss)**: $\approx 0.52 - 0.58$
- Evaluasi divisualisasikan melalui kurva performa epoch dan *Confusion Matrix* menggunakan Seaborn.

---

## 📁 Struktur Direktori

```text
Computer-Vision---Lumpy-Skin/
├── Lumpy-skin-Googlenet.ipynb   # Jupyter Notebook training, evaluasi & visualisasi
├── README.md                    # Dokumentasi proyek
└── (files/ atau dataset)        # Direktori dataset hasil split (train, test, validation)
```

---

## 🚀 Panduan Menjalankan

### 1. Prasyarat Lingkungan
Pastikan Anda memiliki Python 3.8+ dan pustaka berikut terpasang:
```bash
pip install tensorflow opencv-python matplotlib scikit-learn seaborn numpy
```

### 2. Menggunakan Google Colab
Notebook [Lumpy-skin-Googlenet.ipynb](file:///C:/Users/Luthfi/Downloads/Lumpy-Skin/Lumpy-skin-Googlenet.ipynb) telah dioptimalkan untuk dijalankan di Google Colab:
1. Upload file notebook ke Google Colab atau buka dari GitHub.
2. Hubungkan Google Drive yang berisi arsip dataset `lumpy_skin.zip`.
3. Pilih runtime **GPU** (`Runtime > Change runtime type > T4 GPU`).
4. Jalankan seluruh sel dari atas ke bawah.

---

## 🗺️ Rencana Pengembangan (Roadmap)

Rencana pengembangan proyek ke depan disusun ke dalam beberapa tahapan:

### 🔹 Fase 1: Optimasi Model & Eksperimen Arsitektur (Jangka Pendek)
- [ ] **Eksplorasi Backbone Lain**: Membandingkan InceptionV3 dengan arsitektur modern seperti *EfficientNet-B0/B3*, *ConvNeXt*, dan *Vision Transformers (ViT)*.
- [ ] **Integrasi Callbacks**: Menerapkan `EarlyStopping` dan `ModelCheckpoint` untuk mencegah overfitting dan otomatis menyimpan checkpoint bobot terbaik.
- [ ] **Hyperparameter Tuning Otomatis**: Menggunakan *Optuna* atau *Keras Tuner* untuk optimasi learning rate, dropout rate, dan ukuran dense layer.

### 🔹 Fase 2: Peningkatan Kualitas Data & Preprocessing (Jangka Pendek - Menengah)
- [ ] **Ekspansi Dataset**: Mengumpulkan citra tambahan dengan variasi kondisi pencahayaan liar, sudut pandang berbeda, dan berbagai jenis ras sapi.
- [ ] **Advanced Data Augmentation**: Mengintegrasikan library *Albumentations* (CutMix, MixUp, ColorJitter) untuk memperkaya variasi data pelatihan.
- [ ] **Deteksi/Segmentasi Nodul (Object Detection / Segmentation)**:
  - Mengimplementasikan pipeline deteksi objek (misal: YOLOv8 / YOLOv11) untuk melokalisasi posisi nodul sebelum diklasifikasikan.

### 🔹 Fase 3: Explainable AI / Interpretasi Model (Jangka Menengah)
- [ ] **Implementasi Grad-CAM / Grad-CAM++**: Memvisualisasikan heatmap area tubuh sapi yang menjadi fokus keputusan model, membantu dokter hewan memverifikasi apakah model benar-benar fokus pada nodul atau background.

### 🔹 Fase 4: Deployment & Aplikasi Lapangan (Jangka Panjang)
- [ ] **Aplikasi Web Interaktif (Streamlit / Gradio)**: Membangun antarmuka web sederhana agar pengguna dapat mengunggah foto sapi dan melihat hasil prediksi secara instan.
- [ ] **Mobile & Edge Deployment (TFLite / ONNX)**:
  - Mengonversi model ke format `.tflite` dengan kuantisasi (INT8/FP16) agar model ringan (<20 MB) dan dapat berjalan secara offline pada aplikasi smartphone peternak di pedesaan tanpa jaringan internet.
- [ ] **REST API Service**: Membangun backend berbasis FastAPI untuk integrasi ke sistem manajemen peternakan pintar (*Smart Livestock Farming*).

---

## 🤝 Kontribusi

Kontribusi selalu terbuka! Jika Anda memiliki saran, perbaikan dataset, atau ide arsitektur baru:
1. Fork repositori ini
2. Buat branch fitur baru (`git checkout -b fitur-keren`)
3. Commit perubahan Anda (`git commit -m 'Menambahkan fitur keren'`)
4. Push ke branch (`git push origin fitur-keren`)
5. Ajukan Pull Request

---

## 📄 Lisensi

Proyek ini dirilis di bawah lisensi [MIT License](LICENSE).
