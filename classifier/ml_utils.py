import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
from joblib import dump, load
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score,
    roc_curve, auc, f1_score
)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# Path & parameter global
TRAIN_DIR = "./dataset/training"
VAL_DIR = "./dataset/validation"
NEW_DATA_DIR = "./new_data"
EXPECTED_CLASSES = ["01-minor", "02-moderate", "03-severe"]

MODEL_PATH = "cnn_model.h5"
XGB_PATH = "xgboost_model.joblib"

# Fungsi untuk training
def train_models():
    # Data generator
    train_gen = ImageDataGenerator(
        rescale=1./255, rotation_range=15, width_shift_range=0.1,
        height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True
    ).flow_from_directory(TRAIN_DIR, target_size=(128,128), batch_size=16,
                          class_mode="sparse", shuffle=True)

    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        VAL_DIR, target_size=(128,128), batch_size=16,
        class_mode="sparse", shuffle=False
    )

    class_names = list(train_gen.class_indices.keys())
    label_map = {v: k for k, v in train_gen.class_indices.items()}

    # CNN
    inputs = Input(shape=(128,128,3))
    x = Rescaling(1./255)(inputs)
    x = Conv2D(32, (3,3), activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3,3), activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    features = Dense(128, activation="relu")(x)
    feature_model = Model(inputs=inputs, outputs=features)

    # CNN classifier
    x_cls = Dense(len(EXPECTED_CLASSES), activation="softmax")(features)
    full_model = Model(inputs=inputs, outputs=x_cls)
    full_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    full_model.fit(train_gen, validation_data=val_gen, epochs=5,
                   callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)])
    feature_model.save(MODEL_PATH)

    # Ekstraksi fitur
    def extract_features(generator, model):
        X_list, y_list = [], []
        for i in range(len(generator)):
            x_batch, y_batch = generator[i]
            feats = model.predict(x_batch, verbose=0)
            X_list.append(feats); y_list.append(y_batch)
        return np.vstack(X_list), np.hstack(y_list)

    X_train, y_train = extract_features(train_gen, feature_model)
    X_val, y_val = extract_features(val_gen, feature_model)

    # XGBoost
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                            use_label_encoder=False, eval_metric="mlogloss")
    clf.fit(X_train, y_train)
    dump(clf, XGB_PATH)

    # Evaluasi
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=class_names)
    print("✅ Training selesai\n", report)

    # Laporan PDF
    doc = SimpleDocTemplate("Model_Evaluation_Report.pdf", pagesize=A4)
    styles = getSampleStyleSheet(); story = []
    story.append(Paragraph("📊 Model Evaluation Report", styles["Title"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Accuracy: {acc:.4f}", styles["Normal"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Classification Report:", styles["Heading2"]))
    story.append(Paragraph(f"<pre>{report}</pre>", styles["Normal"]))
    doc.build(story)
    return acc, report


# Fungsi prediksi 1 file
def predict_image(img_path):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(XGB_PATH):
        return {"error": "Model belum dilatih"}

    feature_model = tf.keras.models.load_model(MODEL_PATH)
    clf = load(XGB_PATH)

    img = tf.keras.utils.load_img(img_path, target_size=(128,128))
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, 0) / 255.0
    feats = feature_model.predict(arr, verbose=0)
    pred = clf.predict(feats)[0]
    prob = clf.predict_proba(feats).max()
    return {"class": str(pred), "confidence": float(prob)}
