import os
import threading
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from django.shortcuts import render
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
import xgboost as xgb
import cv2
import time
from datetime import datetime
import librosa
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from collections import Counter
from ultralytics import YOLO

# ========================
# Paths & Models
# ========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "classifier/models")
os.makedirs(MODEL_DIR, exist_ok=True)

cnn_path = os.path.join(MODEL_DIR, "cnn_model.h5")
xgb_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
yolo_path = os.path.join(MODEL_DIR, "yolov8_car_damage.pt")  # YOLOv8 trained for car damage

cnn_model = load_model(cnn_path) if os.path.exists(cnn_path) else None
xgb_model = joblib.load(xgb_path) if os.path.exists(xgb_path) else None
yolo_model = YOLO(yolo_path) if os.path.exists(yolo_path) else None

class_names = ["minor", "moderate", "severe"]

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(UPLOAD_DIR, "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ========================
# Status Global
# ========================
upload_status = {
    "processed": 0,
    "total": 0,
    "progress": 0,
    "results": [],
    "metrics": {},
    "scatter_features": [],
    "status": "idle"
}

evaluation_status = {
    "progress": 0,
    "status": "idle",
    "metrics": {},
    "scatter_features": []
}

# ========================
# Pages
# ========================
def upload_page(request):
    return render(request, "classifier/upload.html")

# ========================
# Upload Processing
# ========================
@csrf_exempt
def upload(request):
    global upload_status
    try:
        if request.method != "POST":
            return JsonResponse({"error": "Method not allowed"}, status=405)

        files = request.FILES.getlist("files")
        if not files:
            return JsonResponse({"error": "No files uploaded"}, status=400)

        upload_status = {
            "processed": 0,
            "total": len(files),
            "progress": 0,
            "results": [],
            "status": "processing",
            "metrics": {},
            "scatter_features": []
        }

        y_true = []
        y_pred = []
        features_list = []

        for file in files:
            if file.size > 10 * 1024 * 1024:
                upload_status["results"].append({
                    "filename": file.name,
                    "error": "File terlalu besar, max 10 MB"
                })
                upload_status["processed"] += 1
                upload_status["progress"] = int((upload_status["processed"] / upload_status["total"]) * 100)
                continue

            ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
            filename = f"{ts}_{file.name}"
            file_path = os.path.join(UPLOAD_DIR, filename)

            with open(file_path, "wb") as f:
                for chunk in file.chunks():
                    f.write(chunk)

            ext = os.path.splitext(filename)[1].lower()
            result = {"filename": file.name}

            # === IMAGE ===
            if ext in [".jpg", ".jpeg", ".png"]:
                img_cv = cv2.imread(file_path)
                img_resized = cv2.resize(img_cv, (128, 128))
                img_array = img_resized.astype("float32") / 255.0
                img_array_exp = np.expand_dims(img_array, axis=0)

                # CNN prediction
                cnn_pred = cnn_model.predict(img_array_exp, verbose=0)
                cnn_idx = int(np.argmax(cnn_pred[0]))
                cnn_label = class_names[cnn_idx]
                cnn_conf = float(cnn_pred[0][cnn_idx])

                # XGB prediction
                feat_flat = img_array_exp.flatten().reshape(1, -1)
                xgb_idx = int(xgb_model.predict(feat_flat)[0])
                xgb_label = class_names[xgb_idx]

                # YOLOv8 Object Detection
                if yolo_model:
                    yolo_results = yolo_model.predict(img_cv)
                    for r in yolo_results[0].boxes.data.tolist():
                        x1, y1, x2, y2, score, cls = r
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        label_name = class_names[int(cls)]
                        conf_pct = int(score * 100)
                        color = (0, 255, 0) if label_name=="minor" else (0,165,255) if label_name=="moderate" else (0,0,255)
                        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img_cv, f"{label_name} | {conf_pct}%", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                        cv2.putText(img_cv, f"{label_name} Damage", (x1, y2+25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

                annotated_path = os.path.join(RESULT_DIR, filename)
                cv2.imwrite(annotated_path, img_cv)

                result.update({
                    "file_type": "image",
                    "cnn_prediction": cnn_label,
                    "cnn_confidence": round(cnn_conf, 2),
                    "xgb_prediction": xgb_label,
                    "annotated_path": f"/media/results/{filename}"
                })

                y_true.append(0)
                y_pred.append(xgb_idx)
                features_list.append(feat_flat.flatten())

            # === VIDEO ===
            elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
                cap = cv2.VideoCapture(file_path)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_path = os.path.join(RESULT_DIR, filename)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

                frame_count = 0
                cnn_preds_video = []
                xgb_preds_video = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1

                    if frame_count % 10 == 0:
                        frame_resized = cv2.resize(frame, (128, 128))
                        img_array = frame_resized.astype("float32") / 255.0
                        img_array_exp = np.expand_dims(img_array, axis=0)

                        # CNN
                        cnn_pred = cnn_model.predict(img_array_exp, verbose=0)
                        cnn_idx = int(np.argmax(cnn_pred[0]))
                        cnn_label = class_names[cnn_idx]
                        cnn_preds_video.append(cnn_label)
                        cnn_conf = float(cnn_pred[0][cnn_idx])

                        # XGB
                        feat_flat = img_array_exp.flatten().reshape(1, -1)
                        xgb_idx = int(xgb_model.predict(feat_flat)[0])
                        xgb_label = class_names[xgb_idx]
                        xgb_preds_video.append(xgb_label)

                        # Metrics
                        y_true.append(0)
                        y_pred.append(xgb_idx)
                        features_list.append(feat_flat.flatten())

                        text = f"CNN:{cnn_label}({cnn_conf:.2f}) | XGB:{xgb_label}"
                        cv2.putText(frame, text, (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    (0, 0, 255), 2, cv2.LINE_AA)

                    out.write(frame)

                cap.release()
                out.release()

                result.update({
                    "file_type": "video",
                    "cnn_prediction": Counter(cnn_preds_video).most_common(1)[0][0] if cnn_preds_video else None,
                    "xgb_prediction": Counter(xgb_preds_video).most_common(1)[0][0] if xgb_preds_video else None,
                    "annotated_path": f"/media/results/{filename}"
                })

            # === AUDIO ===
            elif ext == ".wav":
                try:
                    y_audio, sr = librosa.load(file_path, sr=None)
                    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=40)
                    feat = np.mean(mfcc.T, axis=0).reshape(1, -1)

                    xgb_idx = int(xgb_model.predict(feat)[0])
                    xgb_label = class_names[xgb_idx]

                    result.update({
                        "file_type": "audio",
                        "cnn_prediction": None,
                        "cnn_confidence": None,
                        "xgb_prediction": xgb_label,
                        "annotated_path": f"/media/{filename}"
                    })

                    y_true.append(0)
                    y_pred.append(xgb_idx)
                    features_list.append(feat.flatten())
                except Exception as e:
                    result["error"] = f"Audio processing failed: {str(e)}"

            else:
                result["error"] = "Unsupported file type"

            upload_status["results"].append(result)
            upload_status["processed"] += 1
            upload_status["progress"] = int((upload_status["processed"] / upload_status["total"]) * 100)

        # === Metrics & Confusion Matrix ===
        if y_true and y_pred:
            upload_status["metrics"] = {
                "precision": round(precision_score(y_true, y_pred, average='macro'), 4),
                "recall": round(recall_score(y_true, y_pred, average='macro'), 4),
                "f1": round(f1_score(y_true, y_pred, average='macro'), 4),
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
            }

        # === Scatter 3D PCA ===
        if len(features_list) >= 2:
            pca = PCA(n_components=3)
            scatter = pca.fit_transform(np.array(features_list))
            upload_status["scatter_features"] = [
                {"x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2]),
                 "label": class_names[y_pred[i]]} 
                for i, pt in enumerate(scatter)
            ]

        upload_status["status"] = "done"
        return JsonResponse(upload_status)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# ========================
# Progress Endpoint
# ========================
def get_progress(request):
    response = upload_status.copy()
    if not response.get("metrics"):
        response["metrics"] = {
            "confusion_matrix": [[0] * len(class_names) for _ in class_names],
            "precision": 0.0, "recall": 0.0, "f1": 0.0
        }
    if not response.get("scatter_features"):
        response["scatter_features"] = []
    response["progress"] = min(max(response.get("progress", 0), 0), 100)
    return JsonResponse(response)

# ========================
# Download Model
# ========================
def download_model(request, model_type):
    if model_type == "cnn":
        path = cnn_path
    elif model_type == "xgb":
        path = xgb_path
    elif model_type == "yolo":
        path = yolo_path
    else:
        return JsonResponse({"error": "Invalid model type"})
    if os.path.exists(path):
        return FileResponse(open(path, "rb"), as_attachment=True)
    return JsonResponse({"error": "File not found"})
