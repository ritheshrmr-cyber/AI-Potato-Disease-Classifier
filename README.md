# 🌿 Potato Disease Classifier

A Deep Learning web application to detect potato leaf diseases and identify non-potato images.

## 🚀 Features
- Detects:
  - Early Blight
  - Late Blight
  - Healthy Leaves
- Classifies:
  - Non-potato leaf
  - Non-leaf images

## 🛠 Tech Stack
- Python
- TensorFlow / Keras
- FastAPI
- React

## 📂 Structure
- api/ → Backend (FastAPI)
- frontend/ → User Interface
- training.ipynb → Model training

## ▶️ Run

### Backend
```bash
uvicorn api.main:app --reload