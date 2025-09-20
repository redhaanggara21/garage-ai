#!/bin/bash
# =====================================================
# Shell Script: Full Setup Django CNN + XGBoost Project
# =====================================================

# ----------------------------
# 1. Setup Virtual Environment
# ----------------------------
echo "🟢 Membuat virtual environment..."
python3 -m venv venv
source venv/bin/activate

# ----------------------------
# 2. Install Dependencies
# ----------------------------
echo "🟢 Install dependencies..."
pip install --upgrade pip
pip install Django==4.2.12 channels==4.3.0 channels-redis==4.3.0 asgiref==3.9.1 \
celery==5.4.0 redis==5.3.0 tensorflow==2.15.0 numpy==1.25.2 scipy==1.11.2 \
Pillow==10.2.0 h5py==3.9.0 xgboost==1.7.6 matplotlib==3.8.1 seaborn==0.12.2 \
reportlab==4.0.0 python-dotenv==1.0.0 pandas==2.1.1 joblib==1.3.2

# ----------------------------
# 3. Create Django Project
# ----------------------------
echo "🟢 Membuat project Django..."
django-admin startproject myproject .
python manage.py startapp classifier

# ----------------------------
# 4. Create Folders
# ----------------------------
echo "🟢 Membuat folder dataset & new_data..."
mkdir -p dataset/training dataset/validation new_data
mkdir -p classifier/templates

# ----------------------------
# 5. Create Template Files
# ----------------------------
echo "🟢 Membuat template upload.html & train.html..."
cat > classifier/templates/upload.html <<EOL
<!DOCTYPE html>
<html>
<head><title>Upload & Predict</title></head>
<body>
<h1>Upload Image</h1>
<form method="post" enctype="multipart/form-data">{% csrf_token %}{{ form }}<button type="submit">Upload & Predict</button></form>
{% if result %}<h2>Prediction: {{ result.class }} (Confidence: {{ result.confidence|floatformat:2 }})</h2>{% endif %}
<a href="{% url 'train' %}">Go to Training Menu</a>
<div id="progress">Progress: 0%</div>
<script>
let ws = new WebSocket("ws://" + window.location.host + "/ws/progress/");
ws.onmessage = function(e){
    const data = JSON.parse(e.data);
    document.getElementById("progress").innerText = "Progress: " + data.progress + "% | " + data.status;
}
</script>
</body>
</html>
EOL

cat > classifier/templates/train.html <<EOL
<!DOCTYPE html>
<html>
<head><title>Training CNN & XGBoost</title></head>
<body>
<h1>Training Menu</h1>
<form method="post">{% csrf_token %}{{ form }}<button type="submit">Start Training</button></form>
{% if message %}<p>{{ message }}</p>{% endif %}
<a href="{% url 'upload' %}">Back to Upload</a>
<div id="progress">Progress: 0%</div>
<script>
let ws = new WebSocket("ws://" + window.location.host + "/ws/progress/");
ws.onmessage = function(e){
    const data = JSON.parse(e.data);
    document.getElementById("progress").innerText = "Progress: " + data.progress + "% | " + data.status;
}
</script>
</body>
</html>
EOL

# ----------------------------
# 6. Create views.py
# ----------------------------
echo "🟢 Membuat classifier/views.py..."
cat > classifier/views.py <<EOL
from django.shortcuts import render, redirect
from django import forms
from django.conf import settings
from .tasks import train_models_task
import os, pandas as pd, tensorflow as tf, numpy as np, joblib

NEW_DATA_DIR = os.path.join(settings.BASE_DIR, "new_data")
MODEL_CNN = os.path.join(settings.BASE_DIR, "cnn_model.h5")
MODEL_XGB = os.path.join(settings.BASE_DIR, "xgboost_model.joblib")
LABEL_MAP = {0:"01-minor",1:"02-moderate",2:"03-severe"}

class UploadForm(forms.Form):
    image = forms.ImageField()

class TrainForm(forms.Form):
    epochs = forms.IntegerField(initial=10)

def upload_view(request):
    result = None
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data["image"]
            path = os.path.join(NEW_DATA_DIR, img_file.name)
            with open(path, "wb") as f: f.write(img_file.read())
            if os.path.exists(MODEL_CNN) and os.path.exists(MODEL_XGB):
                cnn_model = tf.keras.models.load_model(MODEL_CNN)
                xgb_model = joblib.load(MODEL_XGB)
                arr = tf.keras.utils.img_to_array(tf.keras.utils.load_img(path,target_size=(128,128)))
                arr = np.expand_dims(arr,0)/255.0
                feats = tf.keras.models.Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output).predict(arr)
                pred = xgb_model.predict(feats)[0]
                prob = xgb_model.predict_proba(feats).max()
                result = {"class": LABEL_MAP[int(pred)], "confidence": float(prob)}
    else:
        form = UploadForm()
    return render(request, "upload.html", {"form": form, "result": result})

def train_view(request):
    message = None
    if request.method == "POST":
        form = TrainForm(request.POST)
        if form.is_valid():
            epochs = form.cleaned_data["epochs"]
            train_models_task.delay(os.path.join(settings.BASE_DIR,"dataset/training"),
                                    os.path.join(settings.BASE_DIR,"dataset/validation"),
                                    epochs)
            message = "Training started in background. Check progress below."
    else:
        form = TrainForm()
    return render(request, "train.html", {"form": form, "message": message})
EOL

# ----------------------------
# 7. Create urls.py
# ----------------------------
echo "🟢 Membuat classifier/urls.py & myproject/urls.py..."
cat > classifier/urls.py <<EOL
from django.urls import path
from .views import upload_view, train_view

urlpatterns = [
    path('', upload_view, name='upload'),
    path('train/', train_view, name='train'),
]
EOL

cat > myproject/urls.py <<EOL
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('classifier.urls')),
]
EOL

# ----------------------------
# 8. Create tasks.py
# ----------------------------
echo "🟢 Membuat classifier/tasks.py..."
cat > classifier/tasks.py <<EOL
from celery import shared_task
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
from joblib import dump
import numpy as np
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
import os

CHANNEL_LAYER = get_channel_layer()
IMG_SIZE=(128,128)
BATCH_SIZE=8

def send_progress(progress, status):
    async_to_sync(CHANNEL_LAYER.group_send)(
        "progress_group",
        {"type":"progress_update", "data":{"progress":progress,"status":status}}
    )

@shared_task
def train_models_task(train_dir, val_dir, epochs):
    send_progress(0,"Starting Training")
    train_gen = ImageDataGenerator(rescale=1./255,rotation_range=15,
        width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.1,horizontal_flip=True
    ).flow_from_directory(train_dir,target_size=IMG_SIZE,batch_size=BATCH_SIZE,class_mode="sparse",shuffle=True)
    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(val_dir,target_size=IMG_SIZE,batch_size=BATCH_SIZE,class_mode="sparse",shuffle=False)

    filters=[32,64]
    inputs = Input(shape=IMG_SIZE+(3,))
    x = Rescaling(1./255)(inputs)
    for f in filters:
        x = Conv2D(f,(3,3),activation="relu")(x)
        x = MaxPooling2D()(x)
    x = Flatten()(x)
    features=Dense(128,activation="relu")(x)
    feature_model = Model(inputs=inputs,outputs=features)
    x_cls = Dense(3,activation="softmax")(features)
    full_model = Model(inputs=inputs,outputs=x_cls)
    full_model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

    for epoch in range(epochs):
        full_model.fit(train_gen,validation_data=val_gen,epochs=1)
        send_progress(int((epoch+1)/epochs*100),f"Epoch {epoch+1}/{epochs} completed")

    def extract_features(generator, model):
        X_list,y_list=[],[]
        for i in range(len(generator)):
            x_batch,y_batch=generator[i]
            feats=model.predict(x_batch,verbose=0)
            X_list.append(feats)
            y_list.append(y_batch)
        return np.vstack(X_list), np.hstack(y_list)

    X_train,y_train=extract_features(train_gen,feature_model)
    X_val,y_val=extract_features(val_gen,feature_model)
    clf=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.1,use_label_encoder=False,eval_metric="mlogloss")
    clf.fit(X_train,y_train)
    feature_model.save(os.path.join(os.getcwd(),"cnn_model.h5"))
    dump(clf,os.path.join(os.getcwd(),"xgboost_model.joblib"))
    send_progress(100,"Training Completed")
EOL

# ----------------------------
# 9. Create consumers.py
# ----------------------------
echo "🟢 Membuat classifier/consumers.py..."
cat > classifier/consumers.py <<EOL
import json
from channels.generic.websocket import AsyncWebsocketConsumer

class ProgressConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add("progress_group", self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard("progress_group", self.channel_name)

    async def progress_update(self, event):
        await self.send(text_data=json.dumps(event["data"]))
EOL

# ----------------------------
# 10. Create ASGI for Channels
# ----------------------------
echo "🟢 Membuat myproject/asgi.py..."
cat > myproject/asgi.py <<EOL
import os
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from channels.auth import AuthMiddlewareStack
from django.urls import path
from classifier.consumers import ProgressConsumer

os.environ.setdefault("DJANGO_SETTINGS_MODULE","myproject.settings")

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter([
            path("ws/progress/",ProgressConsumer.as_asgi()),
        ])
    )
})
EOL

# ----------------------------
# 11. Dockerfile & docker-compose
# ----------------------------
echo "🟢 Membuat Dockerfile..."
cat > Dockerfile <<EOL
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip && pip install -r requirements.txt
CMD ["python","manage.py","runserver","0.0.0.0:8000"]
EOL

echo "🟢 Membuat docker-compose.yml..."
cat > docker-compose.yml <<EOL
version: "3.9"
services:
  web:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    depends_on:
      - redis
  redis:
    image: redis:7
EOL

# ----------------------------
# 12. Final Message
# ----------------------------
echo "✅ Shell setup selesai!"
echo "➡️ Masuk ke virtualenv: source venv/bin/activate"
echo "➡️ Jalankan server: python manage.py runserver"
echo "➡️ Atau Docker: docker-compose up --build"
