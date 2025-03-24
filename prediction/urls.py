from django.urls import path
from .views import predict_persona, predict_spending

urlpatterns = [
    path("persona_persona/", predict_persona, name="persona_persona"),
    path("predict_spending/", predict_spending, name="predict_spending"),
]
