import os
import tensorflow as tf
import numpy as np
import joblib
from tensorflow.keras import layers, models
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, accuracy_score, f1_score
from .progress import update_progress

# Global progress state
progress = {"step": 0, "total": 1}

# Cari root project Django
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")  # ✅ dataset di luar app

def update_progress(step, total):
    global progress
    progress["step"] = step
    progress["total"] = total



def train_model(callback=None):
    # Contoh dataset (ganti sesuai dataset kamu)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "data/train", image_size=(128, 128), batch_size=32
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        "data/val", image_size=(128, 128), batch_size=32
    )

    # Normalisasi
    train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

    # CNN sederhana
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(128,128,3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(train_ds.class_names), activation="softmax"),
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=callback)] if callback else None
    )

    return {
        "train_acc": history.history["accuracy"][-1],
        "val_acc": history.history["val_accuracy"][-1],
        "train_loss": history.history["loss"][-1],
        "val_loss": history.history["val_loss"][-1],
    }

def get_progress():
    global progress
    return progress

def save_evaluation_plots(y_true, y_pred, y_prob, class_names):
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("classifier/static/classifier/results/confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("classifier/static/classifier/results/roc_curve.png")
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig("classifier/static/classifier/results/pr_curve.png")
    plt.close()
