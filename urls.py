from django.conf.urls import url,include
from accounts import views
urlpatterns = [
    url(r'^accounts/', include('accounts.urls')),
]
