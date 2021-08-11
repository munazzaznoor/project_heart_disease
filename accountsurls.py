from django.contrib import admin
from django.conf import settings
from accounts import views
from django.conf.urls import url
from django.utils.deprecation import RemovedInDjango40Warning


from django.urls import path
#from . import views
from accounts.views import home,result

urlpatterns = [
    #url(r"^admin/",admin.site.urls),
    #url(r"accounts/", views.home),
    path("",home,name="home"),
    path("result/",result,name="result")
    #url(r"^result",views.result),
]