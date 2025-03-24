from django.contrib import admin
from django.urls import path, include  # Ensure include is imported
from django.http import HttpResponse

def home(request):
    return HttpResponse("Welcome to the Secure Persona Backend!")

urlpatterns = [
    path('', home, name='home'),  # Add this line for the root URL
    path('admin/', admin.site.urls),
    path('prediction/', include('prediction.urls')),  # Include prediction app URLs
]
