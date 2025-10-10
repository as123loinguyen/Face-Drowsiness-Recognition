import os
import io
import base64
import time
import threading
import math
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, request, jsonify
from tensorflow.keras.models import load_model
import mediapipe as mp


app = Flask(__name__)

# ==== Load models ====
emotion_model = load_model("emotional_model.h5", compile=False)
drowsy_model  = load_model("driver_drowsiness_model.keras", compile=False)

print("[INFO] Emotion model:", emotion_model.input_shape, "->", emotion_model.output_shape)
print("[INFO] Drowsy model :", drowsy_model.input_shape,  "->", drowsy_model.output_shape)

# ==== Labels ====
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOJI    = {"Angry": "😡", "Disgust": "🤢", "Fear": "😨",
            "Happy": "😊", "Sad": "😢", "Surprise": "😲", "Neutral": "😐"}
DROWSY   = ["Awake", "Drowsy"]
DROWSY_ICON = {"Awake": "🙂", "Drowsy": "😴"}

# ==== MediaPipe FaceMesh for landmarks ====
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1,
    refine_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---- Shared status for /status polling (UI badges) ----
LAST_STATUS = {
    "ts": 0,
    "emotion_label": "—",
    "emotion_conf": 0.0,
    "drowsy_label": "Awake",
    "drowsy_conf": 0.0,
    "face_detected": False,
}

# ===== Helper: preprocess according to model input shape =====
def preprocess_frame(bgr_img, model):
    H, W, C = model.input_shape[1:4]
    if C == 1:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (W, H))
        arr  = gray.astype("float32") / 255.0
        arr  = np.expand_dims(arr, -1)
    else:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (W, H))
        arr = rgb.astype("float32") / 255.0
    return np.expand_dims(arr, 0)


# ===== Draw helpers for top badges =====
def draw_badge(img, x, y, text, active=False):
    pad = 6
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    w, h = tw + pad * 2, th + pad * 2
    color_bg = (60, 60, 60) if not active else (30, 160, 255)
    cv2.rectangle(img, (x, y), (x + w, y + h), color_bg, -1)
    cv2.putText(img, text, (x + pad, y + h - pad - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return x + w + 8


def draw_header_bar(frame, status):
    # dark strip
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 48), (25, 25, 25), -1)
    cv2.putText(frame, "Driver Monitoring System", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    x = 380
    x = draw_badge(frame, x, 10, "Face Detected", status["face_detected"])
    x = draw_badge(frame, x, 10, "Drowsiness",   status["drowsy_label"] == "Drowsy")
    x = draw_badge(frame, x, 10, "Awake",        status["drowsy_label"] == "Awake")
    x = draw_badge(frame, x, 10, f"Emotion: {status['emotion_label']}", True)


# Eye aspect ratio helper
def eye_aspect_ratio(pts):
    # pts: 6 landmark points of one eye
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C)


def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera 0")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # --- 3D model points for head pose ---
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        status = LAST_STATUS.copy()
        status["face_detected"] = False
        status["yaw"] = status["pitch"] = status["roll"] = None
        status["left_eye"] = status["right_eye"] = "—"

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        if len(faces) > 0:
            status["face_detected"] = True

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        # --- Head pose & eye status from landmarks ---
        if res.multi_face_landmarks:
            h, w = frame.shape[:2]
            lm = res.multi_face_landmarks[0].landmark

            # 2D image points
            image_points = np.array([
                (lm[1].x * w, lm[1].y * h),     # Nose tip
                (lm[152].x * w, lm[152].y * h), # Chin
                (lm[263].x * w, lm[263].y * h), # Left eye left corner
                (lm[33].x * w, lm[33].y * h),   # Right eye right corner
                (lm[287].x * w, lm[287].y * h), # Left mouth corner
                (lm[57].x * w, lm[57].y * h),   # Right mouth corner
            ], dtype=np.float64)

            # Camera internals
            focal_length = w
            center = (w/2, h/2)
            cam_matrix = np.array([[focal_length, 0, center[0]],
                                   [0, focal_length, center[1]],
                                   [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))

            success, rot_vec, trans_vec = cv2.solvePnP(
                MODEL_POINTS, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success:
                rot_mat, _ = cv2.Rodrigues(rot_vec)
                pose_mat = cv2.hconcat((rot_mat, trans_vec))
                _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)
                yaw, pitch, roll = [float(a) for a in euler.flatten()]
                status["yaw"]   = yaw
                status["pitch"] = pitch
                status["roll"]  = roll

            # --- Eye status (EAR threshold) ---
            LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
            RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
            l_pts = np.array([(lm[i].x * w, lm[i].y * h) for i in LEFT_EYE_IDX])
            r_pts = np.array([(lm[i].x * w, lm[i].y * h) for i in RIGHT_EYE_IDX])

            ear_left  = eye_aspect_ratio(l_pts)
            ear_right = eye_aspect_ratio(r_pts)

            status["left_eye"]  = "Closed" if ear_left  < 0.22 else "Open"
            status["right_eye"] = "Closed" if ear_right < 0.22 else "Open"

        # --- Emotion & drowsy detection (giữ nguyên code cũ) ---
        for (x, y, w, h) in faces[:1]:
            roi = frame[y:y+h, x:x+w].copy()
            if roi.size == 0:
                continue
            # Emotion
            x_in_em = preprocess_frame(roi, emotion_model)
            preds_em = emotion_model.predict(x_in_em, verbose=0)[0]
            idx_em   = int(np.argmax(preds_em))
            label_em = EMOTIONS[idx_em]
            conf_em  = float(np.max(preds_em))
            emoji    = EMOJI[label_em]

            # Drowsy
            x_in_dr = preprocess_frame(roi, drowsy_model)
            preds_dr = drowsy_model.predict(x_in_dr, verbose=0)[0]
            idx_dr   = int(np.argmax(preds_dr))
            label_dr = DROWSY[idx_dr]
            conf_dr  = float(np.max(preds_dr))
            icon     = DROWSY_ICON[label_dr]

            status["emotion_label"] = label_em
            status["emotion_conf"]  = conf_em
            status["drowsy_label"]  = label_dr
            status["drowsy_conf"]   = conf_dr

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emoji} {label_em} {conf_em:.2f}",
                        (x, y-45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"{icon} {label_dr} {conf_dr:.2f}",
                        (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            break

        draw_header_bar(frame, status)

        status["ts"] = time.time()
        LAST_STATUS.update(status)

        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    cap.release()


# ==== Home (GET + POST upload image) ====
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index1.html", error="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index1.html", error="No file selected")

        img_bytes = file.read()
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Emotion
        x_in_em = preprocess_frame(img, emotion_model)
        preds_em = emotion_model.predict(x_in_em, verbose=0)[0]
        label_em = EMOTIONS[int(np.argmax(preds_em))]
        conf_em  = float(np.max(preds_em))
        emoji    = EMOJI[label_em]

        # Drowsy
        x_in_dr = preprocess_frame(img, drowsy_model)
        preds_dr = drowsy_model.predict(x_in_dr, verbose=0)[0]
        label_dr = DROWSY[int(np.argmax(preds_dr))]
        conf_dr  = float(np.max(preds_dr))
        icon     = DROWSY_ICON[label_dr]

        ok, buf = cv2.imencode(".jpg", img)
        img_data = base64.b64encode(buf).decode("utf-8") if ok else ""

        return render_template("index1.html",
                               predictions=[
                                   {"class": f"{emoji} {label_em}", "confidence": f"{conf_em:.2f}"},
                                   {"class": f"{icon} {label_dr}",  "confidence": f"{conf_dr:.2f}"}
                               ],
                               img_data=img_data)

    return render_template("index1.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    s = LAST_STATUS.copy()
    return jsonify(s)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF warnings
    app.run(host="0.0.0.0", port=5000, debug=True)
