# 🍚 Rice Quality Classification using MobileNetV2

## 📌 Deskripsi Proyek

Proyek ini bertujuan untuk mengklasifikasikan kualitas beras ke dalam tiga kategori, yaitu:

- Premium
- Medium
- Rendah

Model dibangun menggunakan metode Transfer Learning dengan arsitektur MobileNetV2 yang telah dilatih sebelumnya pada dataset ImageNet. Pendekatan ini dipilih karena mampu menghasilkan performa yang baik dengan waktu pelatihan yang relatif efisien.

---

## 📂 Dataset

Dataset terdiri dari citra beras yang dikelompokkan ke dalam tiga kelas kualitas:

| Kelas | Deskripsi |
|--------|------------|
| Premium | Beras dengan kualitas terbaik |
| Medium | Beras dengan kualitas sedang |
| Rendah | Beras dengan kualitas rendah |

Jumlah data yang digunakan:

| Dataset | Jumlah Gambar |
|----------|--------------|
| Training | 4.500 |
| Validation | 326 |
| Testing | 326 |

---

## 🛠️ Library yang Digunakan

```python
import pandas as pd
import os, shutil
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import random
import pathlib
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')
import glob

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from random import sample
from shutil import copyfile
from pathlib import Path
from tqdm.notebook import tqdm as tq
```

---

## 🤖 Arsitektur Model

Model menggunakan MobileNetV2 sebagai feature extractor dengan konfigurasi berikut:

```python
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])
```

### Konfigurasi Model

| Parameter | Nilai |
|------------|--------|
| Input Size | 224 x 224 |
| Backbone | MobileNetV2 |
| Optimizer | Adam |
| Epoch | 10 |
| Output Activation | Softmax |
| Jumlah Kelas | 3 |

---

## 📈 Hasil Pelatihan

| Training Accuracy| 86.40% 
|---------|------------------|
| Validation Accuracy | 88.65% |

---

## 📊 Evaluasi Model

### Akurasi Pengujian

Model memperoleh akurasi pengujian sebesar:

```text
Accuracy Test : 88%
```

### Classification Report

| Kelas | Precision | Recall | F1-Score |
|---------|----------|---------|---------|
| Medium | 0.90 | 0.95 | 0.93 |
| Premium | 0.81 | 0.98 | 0.89 |
| Rendah | 1.00 | 0.70 | 0.82 |

### Ringkasan Performa

| Metric | Nilai |
|----------|--------|
| Accuracy | 0.88 |
| Macro Average F1-Score | 0.88 |
| Weighted Average F1-Score | 0.88 |

---

## 📌 Kesimpulan

- MobileNetV2 berhasil digunakan untuk mengklasifikasikan kualitas beras ke dalam tiga kategori.
- Model mencapai akurasi pengujian sebesar 88%.
- Kelas Premium memiliki nilai recall tertinggi sebesar 98%, menunjukkan kemampuan model yang sangat baik dalam mengenali beras premium.
- Kelas Rendah memiliki precision sempurna sebesar 100%, namun recall masih dapat ditingkatkan.
- Transfer Learning terbukti efektif dalam menghasilkan model yang akurat dengan waktu pelatihan yang lebih efisien.

---

## 🚀 Pengembangan Selanjutnya

Beberapa pengembangan yang dapat dilakukan:

- Melakukan fine-tuning pada layer MobileNetV2.
- Menambah jumlah dataset untuk meningkatkan generalisasi model.
- Menggunakan teknik augmentasi data yang lebih beragam.
- Mencoba arsitektur lain seperti EfficientNet, ResNet50, atau DenseNet.
- Melakukan deployment model menggunakan Streamlit atau TensorFlow Lite.

---

## 👥 Tim Pengembang

| Nama | Peran |
|--------|--------|
| Sri Ulan Muharromi, S.Kom | Machine Learning Engineer |
| Srie Wahyudhanis Hadis, S.Kom | Machine Learning Engineer |
| Ary Kurnia | Back End Engineer |
| Camelia Regista | Non Technical |
| Yeni | Non Technical |

---

## 📄 Teknologi yang Digunakan

- Python
- TensorFlow / Keras
- MobileNetV2
- Scikit-Learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Google Colab

---

**Project:** Rice Quality Classification  
**Framework:** TensorFlow/Keras  
**Model:** MobileNetV2 Transfer Learning  
**Task:** Image Classification
