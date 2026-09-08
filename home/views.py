# from django.http import HttpResponse


# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")

from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template("home/index.html")
    
    
    # TODO: replace with the real group members before submitting.
    students = [
        {"name": "Muhammad Noor Ullah Ejaz", "matriculation": "672421"},
    ]
    
    projects = [
        {"name": "Project 1: Supervised learning interface",
         "url_name": "project1:index"},
        {"name": "Project 2: Explainability",
         "url_name": "project2:index"},
        {"name": "Project 3: Active learning for learning to defer",
         "url_name": "project3:index"},
         {"name": "Project 4: Preference elicitation",
         "url_name": "project4:index"},
    ]
    
    context = { 
        "students": students, 
        "projects": projects, 
    }
    
    return HttpResponse(template.render(context, request))