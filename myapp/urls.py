# myapp/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('detail', views.detail, name='detail'),
    path('scholar_results/', views.scholar_results, name='scholar_results'),
    path('news/', views.scrape_news, name='scrape_news'),
    path('fetch_news_content/', views.fetch_news_content, name='fetch_news_content'),
]
