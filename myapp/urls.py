from django.contrib import admin
from django.urls import path, include
from myapp import views

urlpatterns = [
    path('', views.index, name='index'),
    path('index/', views.index, name='index'),
    path('about/', views.about, name='about'),
    path('services/', views.services, name='services'),
    # path('<str:url_end>/', views.dynamic_url, name='dynamic_url'),
]