from django.urls import path

from . import views

app_name = "project1"

urlpatterns = [
    path("", views.index, name="index"),
    path("<int:pk>/explore/", views.explore, name="explore"),
    path("<int:pk>/task/", views.set_task, name="set_task"),
    path("<int:pk>/train/", views.train, name="train"),
    path("<int:pk>/delete/", views.delete_dataset, name="delete"),
    path("run/<int:pk>/", views.run, name="run"),
]
