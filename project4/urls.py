from django.urls import path

from . import views

app_name = "project4"

urlpatterns = [
    path("", views.index, name="index"),
    path("report/", views.report, name="report"),
    path("study/", views.consent, name="consent"),
    path("study/start/", views.start, name="start"),
    path("study/task/", views.task, name="task"),
    path("study/ratings/", views.ratings, name="ratings"),
    path("study/final/", views.final, name="final"),
    path("study/debrief/", views.debrief, name="debrief"),
    path("export/", views.export, name="export"),
]
