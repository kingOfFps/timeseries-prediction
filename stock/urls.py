
from django.contrib import admin
from django.urls import path, include
from django.urls import re_path as url

from . import views

urlpatterns = [
    path("admin/", admin.site.urls),
    # path('api-auth/', include('rest_framework.urls')),
    url(r'^', include('stockapp.urls')),
    path('login/', views.user_login, name='login'),
    path('register/', views.user_register, name='register'),
    path('logout/', views.user_logout, name='logout'),
    # path('', views.home, name='home'),
    # path('home/', views.user_logout, name='home'),
    # path('stock_data/', include('stockapp.urls'), name='stock_data'),
]
