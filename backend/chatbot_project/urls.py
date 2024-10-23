from django.contrib import admin
from django.urls import path
from chatbot import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('chat/', views.chatbot_view, name='chat'),  # Route for chat
    path('upload/', views.upload_file, name='upload'),  # Route for dataset upload
]
