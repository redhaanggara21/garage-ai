from celery import shared_task, current_task
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import xgboost as xgb
from sklearn.metrics import accuracy_score

TRAIN_DIR = "./dataset/training"
VAL_DIR = "./dataset/validation"

@shared_task(bind=True)
def train_models(self):
    IMG_SIZE = (64, 64)
    BATCH_SIZE = 8
    EPOCHS = 10

    # Data generator
    train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="sparse", shuffle=True
    )
    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="sparse", shuffle=False
    )

    inputs = Input(shape=IMG_SIZE + (3,))
    x = Rescaling(1./255)(inputs)
    x = Conv2D(16, (3, 3), activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    features = Dense(128, activation="relu")(x)
    x_cls = Dense(len(train_gen.class_indices), activation="softmax")(features)
    model = Model(inputs, x_cls)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}

    # Custom training loop supaya bisa update progress
    for epoch in range(EPOCHS):
        hist = model.fit(train_gen, validation_data=val_gen, epochs=1, verbose=0)
        history["loss"].append(float(hist.history["loss"][0]))
        history["val_loss"].append(float(hist.history["val_loss"][0]))
        history["acc"].append(float(hist.history["accuracy"][0]))
        history["val_acc"].append(float(hist.history["val_accuracy"][0]))

        progress = int(((epoch+1) / EPOCHS) * 100)
        self.update_state(
            state="PROGRESS",
            meta={
                "epoch": epoch+1,
                "total_epochs": EPOCHS,
                "progress": progress,
                "acc": history["acc"][-1],
                "val_acc": history["val_acc"][-1]
            }
        )
        time.sleep(1)  # biar ada jeda keliatan

    # Simpan model
    model.save("cnn_model.h5")

    return {"status": "done", "history": history}

@shared_task(bind=True)
def predict_image(self, file_path):
    import os
    import numpy as np
    import tensorflow as tf
    import joblib

    # Muat model
    feature_model = tf.keras.models.load_model("cnn_model.h5")
    clf = joblib.load("xgboost_model.joblib")

    # Preprocess image
    img = tf.keras.utils.load_img(file_path, target_size=(64, 64))
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, 0) / 255.0

    # Update progress step 1
    self.update_state(state="PROGRESS", meta={"step": 1, "progress": 30})

    feats = feature_model.predict(arr, verbose=0)
    self.update_state(state="PROGRESS", meta={"step": 2, "progress": 70})

    pred = clf.predict(feats)[0]
    prob = clf.predict_proba(feats).max()

    # Update progress step final
    self.update_state(state="PROGRESS", meta={"step": 3, "progress": 90})

    label_map = {0: "01-minor", 1: "02-moderate", 2: "03-severe"}

    result = {
        "class": label_map[int(pred)],
        "confidence": float(prob),
        "file": os.path.basename(file_path)
    }

    return result
