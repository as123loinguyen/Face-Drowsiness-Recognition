# 🧠 Face-Based Drowsiness Recognition

Ứng dụng web **Flask** nhận diện **buồn ngủ** và **cảm xúc khuôn mặt** từ ảnh hoặc webcam theo thời gian thực bằng **Deep Learning**.

---

## 📸 Tổng quan

Hệ thống kết hợp **OpenCV** (thị giác máy tính) và **TensorFlow/Keras** (mô hình học sâu) để:
- Phát hiện trạng thái: **Drowsy / Awake**
- Nhận diện cảm xúc: **Happy, Sad, Neutral, Angry, Surprise, …**

Ứng dụng phù hợp cho:
- Giám sát người lái xe 🚗  
- An toàn lao động 🏭  
- Tương tác người–máy 🧑‍💻

---

## ⚙️ Tính năng

- ✅ Nhận diện **buồn ngủ** từ đặc trưng mắt/biểu cảm  
- ✅ Nhận diện **cảm xúc** khuôn mặt  
- ✅ Hỗ trợ **upload ảnh** và (tuỳ chọn) **webcam real‑time**  
- ✅ Giao diện web Flask đơn giản, dễ dùng  
- ✅ Dễ cài đặt/chạy trên Windows/macOS/Linux

---

## 🏗️ Công nghệ

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python 3.10+ |
| Backend | Flask |
| Deep Learning | TensorFlow / Keras |
| Thị giác máy tính | OpenCV |
| Frontend | HTML, CSS (Bootstrap) |
| Quản lý mã nguồn | Git + GitHub |

---

## 🗂️ Cấu trúc thư mục

```
Face-Drowsiness-Recognition/
│
├── app.py                         # Flask app chính
├── templates/
│   └── index1.html                # Giao diện chính (điều chỉnh theo dự án)
│
├── static/
│   └── uploads/                   # Ảnh vào/ra (nếu có)
│
├── driver_drowsiness_model.keras  # (KHÔNG đưa lên GitHub)
├── emotional_model.h5             # (KHÔNG đưa lên GitHub)
│
├── requirements.txt               # Dependencies
├── .gitignore
└── README.md
```

> 📝 Ghi chú: Trong repo GitHub, **không nên** commit các file model >100MB. Chia sẻ qua Google Drive/HuggingFace và hướng dẫn tải trong README.

---

## 🧠 Model Files

> ⚠️ Do giới hạn 100MB của GitHub, **các model không nằm trong repo**. Hãy tải và đặt cạnh `app.py`.

| File | Mô tả | Link tải |
|---|---|---|
| `driver_drowsiness_model.keras` | Model phát hiện buồn ngủ | *(thay link của bạn ở đây)* `https://drive.google.com/your_keras_link` |
| `emotional_model.h5` | Model nhận diện cảm xúc | https://drive.google.com/file/d/1Ob0sy18c9GsxzSwTtgoHk8X63H8LzC78/view?usp=sharing |

Sau khi tải, **đặt cả hai file ở thư mục gốc** của dự án (cùng cấp với `app.py`).

---

## 🚀 Hướng dẫn chạy (Quickstart)

### Bước 1: Clone repo
```bash
git clone https://github.com/as123loinguyen/Face-Drowsiness-Recognition.git
cd Face-Drowsiness-Recognition
```

### Bước 2: Tạo & kích hoạt môi trường ảo

**Windows**
```bash
python -m venv venv
venv\Scriptsctivate
```

**macOS/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt thư viện
```bash
pip install -r requirements.txt
# Nếu chưa có requirements.txt, có thể:
# pip install flask opencv-python tensorflow numpy
```

### Bước 4: Thêm model
- Tải 2 file model ở phần **Model Files** phía trên  
- Đặt cạnh `app.py`

### Bước 5: Chạy ứng dụng
```bash
python app.py
```

Mở trình duyệt tới:
```
http://127.0.0.1:5000
```

### Bước 6: Dừng server
Nhấn **Ctrl + C** trong terminal.

---

## 🧩 Cách hoạt động (How it works)

1) Người dùng upload ảnh hoặc bật webcam  
2) OpenCV phát hiện/cắt khuôn mặt, tiền xử lý ảnh  
3) Model Keras suy luận:
   - Mắt/trạng thái → **Drowsy / Awake**
   - Biểu cảm → **Emotion**  
4) Flask hiển thị ảnh kèm **khung + nhãn + xác suất**

---

## 🧪 Ví dụ kết quả (minh hoạ)

| Input | Output |
|---|---|
| ![input](https://via.placeholder.com/220x150?text=Input) | ![output](https://via.placeholder.com/220x150?text=Drowsy+%7C+Happy) |

> Thay ảnh thật bằng screenshot từ `static/uploads/` (nếu bạn lưu ảnh kết quả).

---

## 🛠️ Lỗi thường gặp (Troubleshooting)

- **`OSError: ... model not found`**  
  → Chưa đặt `*.keras`/`*.h5` cạnh `app.py`. Kiểm tra lại đường dẫn.

- **Webcam không hoạt động**  
  → Kiểm tra quyền camera của trình duyệt (Site Settings → Camera → Allow) và đóng app khác đang dùng camera (Zoom/Meet).

- **ImportError/ModuleNotFoundError**  
  → Chạy `pip install -r requirements.txt`, hoặc kiểm tra phiên bản Python (khuyến nghị 3.10+).

- **Push lên GitHub bị lỗi >100MB**  
  → Đảm bảo `.gitignore` đã có:
  ```
  *.keras
  *.h5
  static/uploads/
  __pycache__/
  *.pyc
  venv/
  .venv/
  ```
  và **không commit** các file model nặng.

---

## 📄 requirements.txt (gợi ý)

```txt
flask
opencv-python
tensorflow
numpy
```

> Tuỳ môi trường, bạn có thể cần `tensorflow-cpu` thay cho `tensorflow`.

---

## 🧾 License

Dự án phục vụ **mục đích học tập & nghiên cứu**.

---

## ✉️ Liên hệ

**as123loinguyen**  
📧 Email: `doan8540@gmail.com`  
🌐 GitHub: https://github.com/as123loinguyen

---

### Gợi ý nâng cao
- Đưa model lên **Google Drive/HuggingFace** rồi gắn link tải  
- Thêm ảnh screenshot giao diện vào README (mục “Ví dụ kết quả”)  
- Đóng gói chạy thực tế bằng **Docker** / hướng dẫn deploy
