# Transfer Learning untuk Klasifikasi Citra dengan Data Terbatas

Tugas Take-Home Sesi 05: *Advanced CNNs and Transfer Learning*

- **Nama:** `Joshua`
- **NIM:** `<NIM>`

## Ringkasan

Proyek ini membandingkan beberapa strategi training pada klasifikasi 10 ras kucing dan anjing dengan data terbatas (sekitar 2.000 gambar), menggunakan PyTorch dan torchvision. Pertanyaan utama:

> Kapan cukup membekukan (*freeze*) backbone untuk feature extraction, dan kapan fine-tuning mutlak diperlukan?

Yang dibandingkan:

1. Training dari nol (*from scratch*), sebagai baseline.
2. Feature extraction (backbone di-freeze, hanya classifier head dilatih).
3. Progressive fine-tuning (`layer4` di-unfreeze dengan differential learning rate).
4. Uji kontrol data augmentation (dengan vs tanpa augmentation).
5. Arsitektur pembanding berorientasi efisiensi (EfficientNet-B0 vs ResNet-18).

Selain itu notebook memuat failure analysis, perbandingan feature representation sebelum dan sesudah fine-tuning, dan visualisasi Grad-CAM.

## Dataset

- **Oxford-IIIT Pet**, dimuat lewat `torchvision.datasets.OxfordIIITPet` dan diunduh otomatis ke folder `./data` saat notebook dijalankan (ukuran unduhan kurang lebih 800 MB).
- Subset **10 kelas** (5 kucing, 5 anjing). Urutan di bawah adalah label baru 0 sampai 9:

| Label | Ras | Jenis |
|---|---|---|
| 0 | Abyssinian | kucing |
| 1 | Bengal | kucing |
| 2 | Persian | kucing |
| 3 | Russian Blue | kucing |
| 4 | Siamese | kucing |
| 5 | Beagle | anjing |
| 6 | Boxer | anjing |
| 7 | Chihuahua | anjing |
| 8 | Pug | anjing |
| 9 | Samoyed | anjing |

- Split **stratified** 70% train / 15% validation / 15% test dengan `random_state=42`. Split resmi `trainval` dan `test` dari torchvision digabung dulu, lalu dibagi ulang. Test set berisi 300 gambar. Pembagian per gambar tersimpan di `split_info.csv`.
- Pembagian dilakukan pada level indeks sebelum transform apa pun, dan notebook memeriksa bahwa tidak ada gambar yang sama di dua split (tanpa data leakage).

## Protokol eksperimen (sama untuk semua kondisi)

| Komponen | Nilai |
|---|---|
| Arsitektur utama | ResNet-18 (`torchvision.models.resnet18`) |
| Pretrained weights | `ResNet18_Weights.IMAGENET1K_V1` |
| Input | 224 × 224, normalisasi ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) |
| Batch size | 32 |
| Optimizer | AdamW, `weight_decay=1e-2` |
| Loss | `CrossEntropyLoss` |
| Early stopping | monitor validation loss, `patience=5`, maksimum 30 epoch |
| Model yang dipakai | best weights (epoch dengan validation loss terendah) |
| Seed | 42 (Python, NumPy, PyTorch), `cudnn.deterministic=True`, `cudnn.benchmark=False` |
| Augmentation (train saja) | Resize 256, RandomResizedCrop 224 (scale 0.8–1.0), RandomHorizontalFlip, RandomRotation 15°, ColorJitter (brightness 0.2, contrast 0.2) |
| Validation dan test | selalu tanpa augmentation (`weights.transforms()`) |
| Metrik per eksperimen | trainable parameters, learning rate per parameter group, epoch hingga konvergen, durasi training, validation accuracy, test Macro F1 |

## Kondisi eksperimen

| Kode | Kondisi | Deskripsi | Learning rate |
|---|---|---|---|
| B1 | Baseline from scratch | ResNet-18 `weights=None`, semua layer dilatih | 1e-3 |
| B2 | Feature extraction | backbone di-freeze, hanya `fc` baru (`Linear(512, 10)`) dilatih | 1e-3 |
| B3 | Progressive fine-tuning | `layer4` di-unfreeze + `fc` | `layer4`: 1e-4, `fc`: 1e-3 |
| B4 | Kontrol data augmentation | kondisi B2 tanpa augmentation (pembanding: B2 dengan augmentation) | 1e-3 |
| B5 | Arsitektur pembanding | EfficientNet-B0 dengan strategi pemenang antara B2 dan B3 (dipilih lewat validation loss), yaitu fine-tune blok terakhir (`E_B3`) | blok terakhir: 1e-4, classifier: 1e-3 |

Pada B2 dan B3, layer BatchNorm yang di-freeze dipaksa ke mode `eval()` agar running statistics ImageNet tidak ikut berubah.

## Hasil ringkas

Seluruh angka berikut bersumber dari `summary_metrics.csv` dan `tabel_konsolidasi.csv` (seed 42, satu kali run).

| Eksperimen | Trainable params | Epoch terbaik | Waktu (detik) | Val acc | Test acc | Test Macro F1 | Overfit gap* |
|---|---|---|---|---|---|---|---|
| B1 from scratch | 11.181.642 | 20 | 364,1 | 0,6221 | 0,5900 | 0,5946 | 0,0794 |
| B2 feature extraction | 5.130 | 24 | 383,7 | 0,9799 | 0,9600 | 0,9610 | 0,0129 |
| B3 fine-tune `layer4` | 8.398.858 | 4 | 123,0 | 0,9799 | 0,9700 | 0,9701 | 0,0172 |
| B4 B2 tanpa augmentation | 5.130 | 29 | 285,4 | 0,9799 | 0,9600 | 0,9606 | 0,0201 |
| B5 EfficientNet-B0 (fine-tune) | 1.142.202 | 15 | 270,1 | 0,9933 | 0,9833 | 0,9835 | 0,0067 |

\* Overfit gap = train accuracy (tanpa augmentation, setelah training selesai) dikurangi validation accuracy.

Model dengan validation loss terendah adalah B5, yang salah memprediksi 5 dari 300 gambar test (lihat `failure_cases.csv`).

## Struktur repositori

Semua file keluaran ditulis ke direktori kerja notebook.

| File | Isi |
|---|---|
| `main.ipynb` | notebook utama (dijalankan lengkap beserta output sel) |
| `README.md` | dokumen ini |
| `requirements.txt` | daftar dependency |
| laporan (`.pdf` atau `.md`) | laporan analisis |
| `split_info.csv` | pembagian train/val/test per gambar |
| `epoch_log.csv` | log training dan validation per epoch semua eksperimen |
| `summary_metrics.csv` | ringkasan metrik per eksperimen (log mentah) |
| `tabel_konsolidasi.csv` | tabel konsolidasi untuk laporan (C1) |
| `tabel_B5_vs_resnet.csv` | perbandingan EfficientNet-B0 dengan ResNet-18 |
| `failure_cases.csv` | detail gambar test yang salah diprediksi |
| `c5_metrics.json` | metrik pergeseran feature (sebelum dan sesudah fine-tuning) |
| `kurva_loss_B1_B2_B3.png` | kurva training loss vs validation loss |
| `failure_analysis.png` | gambar test yang salah diprediksi beserta confidence |
| `failure_model_input.png` | perbandingan gambar asli dengan input yang dilihat model |
| `gradcam_before_after.png` | Grad-CAM `layer4` sebelum (B2) dan sesudah (B3) fine-tuning |

File weights (`*_best.pt`) tidak disertakan karena ukurannya besar. File tersebut dibuat ulang otomatis saat notebook dijalankan.

## Cara menjalankan

### Google Colab (disarankan)

1. Unggah `main.ipynb` ke Google Colab.
2. Pilih **Runtime → Change runtime type → T4 GPU**.
3. Pilih **Runtime → Run all**. Jalankan sel secara berurutan, karena sel berikutnya bergantung pada variabel sel sebelumnya.
4. Dataset terunduh otomatis. Total training kelima eksperimen memakan sekitar 25 menit di GPU T4 (berdasarkan `train_time_sec` di log).

### Lokal

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook main.ipynb
```

GPU CUDA disarankan. Di CPU, training akan jauh lebih lambat. Sel terakhir (export) memakai `google.colab.files` sehingga hanya berfungsi di Colab. Sel tersebut bisa dilewati saat dijalankan lokal.

## Reproducibility

- Seed tetap (42) di-set ulang di awal setiap eksperimen, sehingga hasil tidak bergantung pada urutan menjalankan.
- Hasil bisa sedikit berbeda antar versi PyTorch dan jenis GPU walau `cudnn.deterministic=True`.
- Notebook membaca atribut internal `ds._images` dan `ds._labels` dari `OxfordIIITPet`, yang bisa berubah antar versi torchvision. Karena itu, pin versi persis yang dipakai. Di Colab, jalankan sel berikut lalu salin hasilnya ke `requirements.txt`:

```python
import torch, torchvision, numpy, pandas, sklearn, matplotlib, PIL
for m in (torch, torchvision, numpy, pandas, sklearn, matplotlib, PIL):
    print(f"{m.__name__}=={m.__version__}")
```

## Limitasi

- Semua eksperimen memakai **satu seed** (42). Selisih kecil antar kondisi (misalnya B2 vs B3 vs B4) setara beberapa gambar pada test set 300 gambar dan tidak bisa disimpulkan signifikan secara statistik.
- Hanya **10 kelas** dan sekitar 2.000 gambar, sehingga hasil belum tentu berlaku pada dataset yang lebih besar atau domain yang berbeda.
- B4 hanya diuji pada kondisi B2.
- B5 memakai preprocessing validation/test milik ResNet-18 (`weights.transforms()` ResNet), bukan preset bawaan EfficientNet-B0.
- Sampel failure analysis dipilih dari kesalahan dengan confidence tertinggi, sehingga tidak mewakili semua jenis kesalahan.
- Anggaran komputasi terbatas pada GPU T4 Colab gratis.

## Referensi

- Materi kuliah Sesi 05: Advanced CNNs and Transfer Learning (Slide 5-30).
- Kapoor, A., Gulli, A., Pal, S., Chollet, F. (2022). Deep Learning with TensorFlow and Keras. Packt Publishing.
- Li et al. (2019), "Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift".

