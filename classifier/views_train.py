import os
import threading
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from django.shortcuts import render
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
import xgboost as xgb

# ========================
# Paths & Models
# ========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "classifier/models")
os.makedirs(MODEL_DIR, exist_ok=True)

cnn_path = os.path.join(MODEL_DIR, "cnn_model.h5")
xgb_path = os.path.join(MODEL_DIR, "xgb_model.pkl")

cnn_model = load_model(cnn_path) if os.path.exists(cnn_path) else None
xgb_model = joblib.load(xgb_path) if os.path.exists(xgb_path) else None
class_names = ["minor","moderate","severe"]

training_status = {"epoch":0, "total_epochs":0, "loss":0.0, "accuracy":0.0,
                   "progress":0, "status":"idle", "metrics":{}, "scatter_features":[]}
stop_training_flag = False

# ========================
# Training
# ========================
def update_training_progress(**kwargs):
    global training_status
    training_status.update(kwargs)
    epoch = training_status.get("epoch",0)
    total = training_status.get("total_epochs",1)
    training_status["progress"] = int(epoch/total*100)

@csrf_exempt
def start_training(request):
    global stop_training_flag, cnn_model, xgb_model
    if training_status["status"]=="training":
        return JsonResponse({"message":"Training sedang berjalan!"})
    
    stop_training_flag = False
    training_status.update({"status":"training","epoch":0,"total_epochs":5})
    
    train_dir = os.path.join(BASE_DIR,"dataset/training")
    val_dir = os.path.join(BASE_DIR,"dataset/validation")
    
    def train_thread():
        global cnn_model, xgb_model
        try:
            train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=(128,128), batch_size=16)
            val_ds   = tf.keras.utils.image_dataset_from_directory(val_dir, image_size=(128,128), batch_size=16)
            
            # Simple CNN
            cnn_model = tf.keras.Sequential([
                tf.keras.layers.Rescaling(1./255, input_shape=(128,128,3)),
                tf.keras.layers.Conv2D(32,3,activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64,3,activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(name="flatten_layer"),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(len(train_ds.class_names), activation='softmax')
            ])
            cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            embedding_model = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer("flatten_layer").output)
            
            for epoch in range(5):
                if stop_training_flag:
                    training_status["status"]="stopped"
                    return
                history = cnn_model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=0)
                training_status["epoch"]=epoch+1
                training_status["loss"]=history.history['loss'][0]
                training_status["accuracy"]=history.history['accuracy'][0]
                
                # Confusion matrix
                y_true, y_pred = [], []
                for imgs, labels in val_ds:
                    preds = np.argmax(cnn_model.predict(imgs,verbose=0),axis=1)
                    y_true.extend(labels.numpy())
                    y_pred.extend(preds)
                training_status["metrics"] = {
                    "precision": round(precision_score(y_true,y_pred,average='macro',labels=list(range(len(class_names))),zero_division=0),4),
                    "recall": round(recall_score(y_true,y_pred,average='macro',labels=list(range(len(class_names))),zero_division=0),4),
                    "f1": round(f1_score(y_true,y_pred,average='macro',labels=list(range(len(class_names))),zero_division=0),4),
                    "confusion_matrix": confusion_matrix(y_true,y_pred,labels=list(range(len(class_names)))).tolist()
                }
                
                # Scatter PCA
                embeddings, labels_list = [], []
                for imgs, lbls in train_ds:
                    emb = embedding_model.predict(imgs, verbose=0)
                    embeddings.append(emb)
                    labels_list.extend(lbls.numpy())
                embeddings = np.vstack(embeddings)
                n_samples, n_features = embeddings.shape
                n_components = min(3, n_samples, n_features)
                if n_components > 0:
                    scatter_3d = PCA(n_components=n_components).fit_transform(embeddings)
                else:
                    scatter_3d = np.zeros((n_samples,3))
                training_status["scatter_features"] = [
                    {"x": float(pt[0]), "y": float(pt[1]) if n_components>1 else 0.0,
                     "z": float(pt[2]) if n_components>2 else 0.0,
                     "label": class_names[labels_list[i]]}
                    for i, pt in enumerate(scatter_3d)
                ]
            
            cnn_model.save(cnn_path)
            
            # Dummy XGBoost
            X_train = np.random.rand(100, 128*128*3)
            y_train = np.random.randint(0,len(class_names),100)
            xgb_model = xgb.XGBClassifier()
            xgb_model.fit(X_train, y_train)
            joblib.dump(xgb_model, xgb_path)
            
            training_status["status"]="done"
        except Exception as e:
            training_status["status"]="error"
            print("Error saat training:", e)
    
    threading.Thread(target=train_thread).start()
    return JsonResponse({"message":"Training dimulai!"})

@csrf_exempt
def stop_training(request):
    global stop_training_flag
    stop_training_flag = True
    return JsonResponse({"message":"Stop training dikirim!"})

# ========================
# Unified Progress Endpoint
# ========================
def get_progress(request):
    # progress_type = request.GET.get("type","training").lower()
    # if progress_type=="upload":
    #     status = upload_status
    # elif progress_type=="evaluation":
    #     status = evaluation_status
    # else:
    #     status = training_status
    status = training_status
    response = status.copy()
    if "metrics" not in response or not response["metrics"]:
        response["metrics"] = {
            "confusion_matrix":[[0]*len(class_names) for _ in class_names],
            "precision":0.0,"recall":0.0,"f1":0.0
        }
    if "scatter_features" not in response or not response["scatter_features"]:
        response["scatter_features"]=[]
    response["progress"]=min(max(response.get("progress",0),0),100)
    return JsonResponse(response)

# ========================
# Download Model
# ========================
def download_model(request, model_type):
    if model_type=="cnn":
        path = cnn_path
    elif model_type=="xgb":
        path = xgb_path
    else:
        return JsonResponse({"error":"Invalid model type"})
    if os.path.exists(path):
        return FileResponse(open(path,"rb"), as_attachment=True)
    return JsonResponse({"error":"File not found"})