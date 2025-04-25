# Deepfake Video Detection using Visual and Audio Features

This project is a web-based tool for detecting deepfake videos using both **visual** and **audio (NLP)** features. It uses **XGBoost** as the main model and supports other optional models like Logistic Regression and Random Forest for comparison.

## 🌟 Features

- 🎥 Upload any video file (.mp4)
- 📊 Extracts frame-level visual features (color, texture, motion)
- 🎧 Extracts audio features using NLP-like methods (MFCC, chroma, spectral)
- 🤖 Predicts if the video is FAKE or REAL using XGBoost
- ✅ Displays prediction confidence, extracted features, and visual output
- 🖥️ Clean and responsive frontend UI

---

## 🧠 How It Works

### 📌 1. Video Processing
- Videos are uploaded from the frontend and sent to the FastAPI backend.
- Frames are extracted using OpenCV.
- Audio is extracted using `librosa`.

### 📌 2. Feature Extraction

#### Visual Features:
- Mean and variance of RGB channels
- Edge density using Canny
- Texture entropy using Laplacian
- Motion consistency & smoothness

#### Audio Features (NLP-inspired):
- **MFCCs (Mel-frequency cepstral coefficients)**
- **Spectral centroid**
- **Zero-crossing rate**
- **Chroma features**
- **RMS (Root Mean Square) energy**

These features simulate speech-based consistency and stability often seen in deepfakes.

### 📌 3. Classification (Model)
- Trained using `XGBoost`, a powerful gradient boosting algorithm
- Also supports Logistic Regression and Random Forest for experimentation
- Output includes:
  - Prediction (`REAL` or `FAKE`)
  - Confidence score
  - Feature list

---

## 📁 Project Structure

deepfake-detection/
├── backend/
│   ├── main.py                  # FastAPI backend
│   ├── model/
│   │   ├── xgboost_model.pkl    # Trained XGBoost model
│   │   └── train_model.py       # Training script
│   ├── utils/
│   │   ├── extract_visual.py    # Extract visual features
│   │   └── extract_audio.py     # Extract NLP audio features
├── frontend/
│   ├── index.html               # UI
│   ├── style.css                # Styling
│   └── script.js                # Logic for upload and results
├── train_sample_model/
│   ├── metadata.json            # Labels for training data
│   └── video files              # Deepfake and real videos

---

## 🛠️ Requirements

### Python Packages (backend)

Install with:

```bash
pip install -r requirements.txt
```

#### `requirements.txt`

fastapi
uvicorn
xgboost
scikit-learn
opencv-python
numpy
pandas
librosa
pydub
python-multipart

> Ensure `ffmpeg` is installed for audio extraction:

```bash
sudo apt install ffmpeg
```

---

## 🚀 Run the Project

### Backend (FastAPI)

```bash
cd backend
uvicorn main:app --reload
```

### Frontend

Just open `frontend/index.html` in a browser (you can also serve via a static server).

---

## 🧪 Training the Model

To retrain the model:

```bash
python backend/model/train_model.py
```

This script reads labeled videos from `train_sample_model/`, extracts visual and audio features, and trains the XGBoost model (`xgboost_model.pkl`).

---

## 🧠 Models Used

- ✅ **XGBoost**: Primary classifier
- 🔍 Logistic Regression & Random Forest (optional comparison)
- 🎧 **NLP-inspired audio features** via `librosa`

---

## 📸 Sample Output

- **Prediction**: REAL
- **Confidence**: 92.85%
- **Extracted Features**: Displayed in a clean grid UI
- **Color-coded**: REAL (green), FAKE (red)

---

## 📚 Credits

Created by Ashwin Sujesh  
[GitHub](https://github.com/Ash2004win) | [LinkedIn](https://www.linkedin.com/in/ashwin-sujesh)
