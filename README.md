# 🌿 AI-Powered Plant Disease Detection System

A deep learning-based web application that detects plant diseases from leaf images using **MobileNetV2 (Transfer Learning)** and provides real-time predictions with treatment recommendations via a **Streamlit UI**.

---

## 📸 Features

- 🧠 MobileNetV2 (pretrained on ImageNet)
- 📷 Upload leaf image for real-time prediction
- 🎯 Predicts: Healthy · Powdery · Rust
- 📊 Confidence score + probability breakdown
- 💊 Disease description, symptoms, treatment & prevention
- 📈 Model performance dashboard (accuracy, loss, confusion matrix)
- ⚡ Fast inference (~50–150ms)

---

## 📁 Project Structure

plant-disease-detection-ai/
├── app.py
├── train.py
├── predict.py
├── requirements.txt
│
├── models/
│ ├── class_labels.json
│ └── training_history.json
│
├── assets/
│ ├── training_curves.png
│ └── confusion_matrix.png

---

## 📦 Dataset

Dataset used: **Plant Village Dataset (Kaggle)**

🔗 Link: https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset

### Classes:
- Healthy
- Powdery (Powdery Mildew)
- Rust

### After downloading, structure should be:
Dataset/
├── Train/Train/{Healthy,Powdery,Rust}/
├── Validation/Validation/{Healthy,Powdery,Rust}/
└── Test/Test/{Healthy,Powdery,Rust}/

---

## ⚠️ Model File Note

The trained model (`.h5`) is **not included** due to GitHub size limitations.

To generate the model locally:

python train.py

🚀 How to Run
1. Clone Repository
git clone https://github.com/yourusername/plant-disease-detection-ai.git
cd plant-disease-detection-ai
2. Install Dependencies
pip install -r requirements.txt
3. Train the Model
python train.py
4. Run the App
streamlit run app.py
Open in browser:
http://localhost:8501
