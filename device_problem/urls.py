from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^training/(?P<end_report_key>\d+)$', views.training, name="training"),
    url('status/', views.status, name="status")
]
