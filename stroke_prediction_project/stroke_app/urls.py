from django.urls import path
from . import views
from .views import upload_csv_and_predict, success_view, predict_image_stroke
from django.contrib import admin
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.predict_stroke, name='predict_stroke'),
    path('upload/', upload_csv_and_predict, name='upload'),
    path('success/', success_view, name='success_page'),
    path('upload_image/', predict_image_stroke, name='upload_image'),
    path('reports/', views.stroke_report, name='stroke_report'),
    # Ruta para la página de administrador
    path('admin/', admin.site.urls),
    # URL para logout
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),
]