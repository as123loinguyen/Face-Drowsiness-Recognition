# 🧠 Face-Based Drowsiness Recognition

A Flask-based web application that detects **drowsiness** and **facial emotions** from images or webcam in real-time using Deep Learning.

---

## 📸 Overview

This project uses **Computer Vision (OpenCV)** and **Deep Learning (Keras/TensorFlow)** to detect whether a person is **drowsy** or **awake**, and simultaneously recognize their **facial emotion** (happy, sad, neutral, angry, etc.).

It can be applied in:
- Driver monitoring systems 🚗
- Workplace safety systems 🏭
- Human–computer interaction 🧑‍💻

---

## ⚙️ Features

✅ Detect **drowsiness** (eyes closed / open)  
✅ Recognize **emotion** (happy, sad, neutral, angry, surprise, etc.)  
✅ Support **image upload** and optional **real-time webcam**  
✅ Web-based interface (Flask + HTML/CSS/JS)  
✅ Portable and lightweight  

---

## 🏗️ Tech Stack

| Component | Technology |
|------------|-------------|
| Programming Language | Python 3.10 |
| Framework | Flask |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Frontend | HTML, CSS, Bootstrap |
| Version Control | Git + GitHub |

---

## 🗂️ Project Structure

Face-Drowsiness-Recognition/
│
├── app.py # Main Flask application
├── templates/
│ ├── index.html # Home page
│ └── result.html # Result display page
│
├── static/
│ ├── uploads/ # Uploaded images
│ ├── css/ # Stylesheets
│ └── js/ # Optional JS files
│
├── models/
│ ├── driver_drowsiness_model.keras
│ └── emotional_model.h5
│
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md

yaml
Copy code

---

## 🧠 Model Files

> ⚠️ Due to GitHub’s 100 MB file size limit, the trained model files are **not included** in this repository.

Please download the models from Google Drive and place them in the **root folder** of the project.

| File Name | Description | Download Link |
|------------|--------------|----------------|
| `driver_drowsiness_model.keras` | CNN model for detecting drowsiness | [Download here](https://drive.google.com/your_keras_link) |
| `emotional_model.h5` | CNN model for facial emotion recognition | [Download here](https://drive.google.com/your_h5_link) |

---

## 🚀 How to Run the App

### 1️⃣ Clone the repository
```bash
git clone https://github.com/as123loinguyen/Face-Drowsiness-Recognition.git
cd Face-Drowsiness-Recognition
2️⃣ Create and activate a virtual environment
bash
Copy code
python -m venv venv
venv\Scripts\activate       # For Windows
# or
source venv/bin/activate    # For macOS/Linux
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Download and place model files
Download both .keras and .h5 models from Google Drive
→ place them in the root directory (same folder as app.py)

5️⃣ Run the Flask app
bash
Copy code
python app.py
6️⃣ Open in browser
Go to 👉 http://127.0.0.1:5000

🧩 How It Works
User uploads an image or enables the webcam.

The system detects a face using OpenCV.

Pre-trained models classify:

Eye state → Drowsy / Awake

Facial expression → Emotion type

The result (image + prediction) is displayed on the web interface.

🧪 Example Results
Input	Output

🧾 License
This project is for educational and research purposes only.
Developed by as123loinguyen (2025).

📬 Contact
For questions or collaboration:
📧 doan8540@gmail.com
🌐 GitHub: as123loinguyen
