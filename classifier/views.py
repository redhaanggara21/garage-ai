import os
import threading
import joblib
from sklearn.decomposition import PCA
from django.shortcuts import render
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt


# ========================
# Views
# ========================
def home_view(request):
    return render(request, "classifier/home.html")

def train_page(request):
    return render(request, "classifier/train.html")

def upload_page(request):
    return render(request, "classifier/upload.html")

def kfold_page(request):
    return render(request, "classifier/kfold_evaluate.html")


