from django.urls import path
from . import views
from . import views_kfold_evaluate
from . import views_train
from . import views_upload

urlpatterns = [
    path('', views.home_view, name='home'),
    path('train', views.train_page, name='train_page'),
    path('upload-page/', views_upload.upload_page, name='upload_page'),
    path('upload-process/', views_upload.upload, name='upload-process'),
    path('start-training/', views_train.start_training, name='start_training'),
    path('stop-training/', views_train.stop_training, name='stop_training'),
    path('start-evaluation/', views_kfold_evaluate.start_kfold_evaluate, name='start_evaluation'),
    path('get-progress-training/', views_train.get_progress, name='get_progress_training'),
    path('get-progress-upload/', views_upload.get_progress, name='get_progress_upload'),
    path('get-progress-evaluate/', views_upload.get_progress, name='get_progress_upload'),
    path('download/<str:model_type>/', views_train.download_model, name='download_model'),
    path('kfold/', views.kfold_page, name='kfold_page'),
    path('kfold/start/', views_kfold_evaluate.start_kfold_evaluate, name='start_kfold_evaluate'),
    path('kfold/progress/', views_kfold_evaluate.get_kfold_progress, name='get_kfold_progress')
]
