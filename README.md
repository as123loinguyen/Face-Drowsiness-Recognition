# 🧠 Face-Based Drowsiness Recognition

A Flask-based web application that detects **drowsiness** and **facial emotions** from images or webcam in real time using **Deep Learning**.

---

## 📸 Overview

This project combines **OpenCV** (computer vision) and **TensorFlow/Keras** (deep learning) to:
- Classify eye/face state: **Drowsy / Awake**
- Recognize facial emotions: **Happy, Sad, Neutral, Angry, Surprise, …**

Use cases:
- Driver monitoring systems 🚗  
- Workplace safety 🏭  
- Human–computer interaction 🧑‍💻

---

## ⚙️ Features

- ✅ Drowsiness detection from facial/eye cues  
- ✅ Facial emotion recognition  
- ✅ Image upload and optional real-time webcam  
- ✅ Simple Flask web UI (HTML/CSS/JS)  
- ✅ Easy setup on Windows/macOS/Linux

---

## 🏗️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Backend | Flask |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Frontend | HTML, CSS (Bootstrap) |
| VCS | Git + GitHub |

---

## 🗂️ Project Structure
```
Face-Drowsiness-Recognition/
│
├── app.py                         # Main Flask app
├── templates/
│   └── index1.html                # UI template (rename as needed)
│
├── static/
│   └── uploads/                   # Input/output images (optional)
│
├── driver_drowsiness_model.keras  # (Do NOT commit to GitHub)
├── emotional_model.h5             # (Do NOT commit to GitHub)
│
├── requirements.txt               # Dependencies
├── .gitignore
└── README.md
```

> Note: Do **not** commit trained model files (>100 MB) to GitHub. Share via Google Drive/HuggingFace and reference them here.

---

## 🧠 Model Files

> ⚠️ Due to GitHub’s 100 MB limit, trained models are **not included**. Download and place them next to `app.py`.

| File | Description | Download Link |
|---|---|---|
| `driver_drowsiness_model.keras` | Drowsiness detection model | https://drive.google.com/file/d/1tPQBZZKzcxYON1p3z5SCcgvq1LUXMPMW/view?usp=sharing |
| `emotional_model.h5` | Emotion recognition model | https://drive.google.com/file/d/1Ob0sy18c9GsxzSwTtgoHk8X63H8LzC78/view?usp=sharing |

After downloading, put both files in the **project root** (same level as `app.py`).

---

## 🚀 How to Run the App

### Step 1: Clone the repository
```bash
git clone https://github.com/as123loinguyen/Face-Drowsiness-Recognition.git
cd Face-Drowsiness-Recognition
```

### Step 2: Create and activate a virtual environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
# If requirements.txt is missing, you can use:
# pip install flask opencv-python tensorflow numpy
```

### Step 4: Add the model files
- Download the two model files (see **Model Files** above)  
- Place them next to `app.py`

### Step 5: Run the Flask app
```bash
python app.py
```

Open your browser at:
```
http://127.0.0.1:5000
```

### Step 6: Stop the server
Press **Ctrl + C** in the terminal.

---

## 🧩 How It Works

1. User uploads an image or enables the webcam  
2. OpenCV detects/crops face(s) and pre-processes the frame  
3. Keras models infer:
   - Eye/face state → **Drowsy / Awake**  
   - Facial expression → **Emotion**  
4. Flask returns the annotated image with **boxes + labels + scores**

---

## 🧪 Example Results

| Input | Output |
|---|---|
| ![input](https://via.placeholder.com/220x150?text=Input) | ![output](https://via.placeholder.com/220x150?text=Drowsy+%7C+Happy) |

> Replace placeholders with real screenshots from `static/uploads/` (if you save results).

---

## 🛠️ Troubleshooting

- **`OSError: ... model not found`**  
  Ensure `*.keras` / `*.h5` exist next to `app.py` and the paths match.

- **Webcam not working**  
  Check browser camera permissions (Site Settings → Camera → Allow). Close apps using the camera (Zoom/Meet).

- **ImportError / ModuleNotFoundError**  
  Run `pip install -r requirements.txt`, and verify Python version (3.10+ recommended).

- **Push to GitHub fails (>100 MB)**  
  Ensure `.gitignore` includes:
  ```
  *.keras
  *.h5
  static/uploads/
  __pycache__/
  *.pyc
  venv/
  .venv/
  ```
  and **do not commit** large model files.

---

## 📄 requirements.txt (suggested)

```txt
flask
opencv-python
tensorflow
numpy
```

> Depending on your environment, you may prefer `tensorflow-cpu`.

---

## 🧾 License

For **educational and research** purposes.

---

## 📬 Contact

**as123loinguyen**  
Email: `doan8540@gmail.com`  
GitHub: https://github.com/as123loinguyen
