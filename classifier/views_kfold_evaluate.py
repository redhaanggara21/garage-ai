import os
import threading
import numpy as np
import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, log_loss
from sklearn.decomposition import PCA

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "classifier/models")
cnn_path = os.path.join(MODEL_DIR, "cnn_model.h5")
xgb_path = os.path.join(MODEL_DIR, "xgb_model.pkl")

# Load models
cnn_model = load_model(cnn_path) if os.path.exists(cnn_path) else None
xgb_model = joblib.load(xgb_path) if os.path.exists(xgb_path) else None
class_names = ["minor","moderate","severe"]

# Status dict
kfold_status = {"progress":0, "processed":0, "total":0, "results":[], "metrics":{}, "scatter_features":[], "status":"idle"}

# Page
def kfold_evaluate_page(request):
    return render(request, "classifier/kfold_evaluate.html")

# Start K-Fold evaluation
@csrf_exempt
def start_kfold_evaluate(request):
    global kfold_status
    k = int(request.GET.get("k",5))
    epochs = int(request.GET.get("epochs",3))
    upload_dir = os.path.join(BASE_DIR,"uploads")
    files = [f for f in os.listdir(upload_dir) if f.lower().endswith((".png",".jpg",".jpeg"))]
    if not files:
        return JsonResponse({"error":"No uploaded images found"})
    
    kfold_status.update({"progress":0, "processed":0, "total":len(files), "results":[],"metrics":{}, "scatter_features":[],"status":"running"})
    
    def kfold_thread():
        global kfold_status
        X, y_true, filenames = [], [], []
        for f in files:
            img_path = os.path.join(upload_dir, f)
            img_pil = image.load_img(img_path, target_size=(128,128))
            img_array = image.img_to_array(img_pil)
            if img_array.shape[2] == 1: img_array = np.repeat(img_array, 3, axis=2)
            img_array = img_array / 255.0
            X.append(img_array)
            cnn_pred = cnn_model.predict(np.expand_dims(img_array,0), verbose=0)
            y_true.append(int(np.argmax(cnn_pred[0])))
            filenames.append(f)
        X = np.array(X)
        y_true = np.array(y_true)
        y_cat = to_categorical(y_true, num_classes=len(class_names))
        
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        per_fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_cat[train_idx], y_cat[test_idx]
            y_test_labels = y_true[test_idx]

            # Clone CNN model per fold
            fold_model = load_model(cnn_path)
            fold_model.compile(optimizer=Adam(1e-4), loss=CategoricalCrossentropy(), metrics=['accuracy'])
            history = fold_model.fit(X_train, y_train, epochs=epochs, batch_size=8, verbose=0)

            cnn_preds = fold_model.predict(X_test, verbose=0)
            y_cnn_pred_labels = np.argmax(cnn_preds, axis=1)

            y_xgb_pred, y_xgb_prob = [], []
            X_test_flat = X_test.reshape(len(X_test), -1)
            for i in range(len(X_test_flat)):
                pred = int(xgb_model.predict(X_test_flat[i].reshape(1,-1))[0])
                y_xgb_pred.append(pred)
                y_xgb_prob.append(xgb_model.predict_proba(X_test_flat[i].reshape(1,-1))[0])
                kfold_status["results"].append({
                    "filename": filenames[test_idx[i]],
                    "fold": fold_idx+1,
                    "true_label": class_names[y_test_labels[i]],
                    "cnn_pred": class_names[y_cnn_pred_labels[i]],
                    "xgb_pred": class_names[pred]
                })

            per_fold_metrics.append({
                "fold": fold_idx+1,
                "cnn_accuracy": round(accuracy_score(y_test_labels, y_cnn_pred_labels),4),
                "cnn_loss": round(history.history['loss'][-1],4),
                "cnn_precision": round(precision_score(y_test_labels, y_cnn_pred_labels,average='macro', zero_division=0),4),
                "cnn_recall": round(recall_score(y_test_labels, y_cnn_pred_labels,average='macro', zero_division=0),4),
                "cnn_f1": round(f1_score(y_test_labels, y_cnn_pred_labels,average='macro', zero_division=0),4),
                "cnn_confusion_matrix": confusion_matrix(y_test_labels, y_cnn_pred_labels).tolist(),
                "xgb_accuracy": round(accuracy_score(y_test_labels, y_xgb_pred),4),
                "xgb_loss": round(log_loss(y_test_labels, y_xgb_prob, labels=[0,1,2]),4),
                "xgb_precision": round(precision_score(y_test_labels, y_xgb_pred,average='macro', zero_division=0),4),
                "xgb_recall": round(recall_score(y_test_labels, y_xgb_pred,average='macro', zero_division=0),4),
                "xgb_f1": round(f1_score(y_test_labels, y_xgb_pred,average='macro', zero_division=0),4),
                "xgb_confusion_matrix": confusion_matrix(y_test_labels, y_xgb_pred).tolist()
            })

            kfold_status["progress"] = int((fold_idx+1)/k*100)
            kfold_status["processed"] = len(test_idx)*(fold_idx+1)

        # 3D scatter
        pca = PCA(n_components=3)
        X_flat = X.reshape(len(X), -1)
        scatter = pca.fit_transform(X_flat)
        kfold_status["scatter_features"] = [{"x":float(pt[0]),"y":float(pt[1]),"z":float(pt[2]),"label":class_names[y_true[i]]} for i, pt in enumerate(scatter)]
        kfold_status["metrics"] = {"per_fold": per_fold_metrics}
        kfold_status["status"] = "done"

    threading.Thread(target=kfold_thread).start()
    return JsonResponse({"message":"K-Fold evaluation started"})

# Poll K-Fold progress
def get_kfold_progress(request):
    return JsonResponse(kfold_status)
