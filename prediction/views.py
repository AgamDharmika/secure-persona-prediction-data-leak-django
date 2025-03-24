from django.shortcuts import render

# Create your views here.
import logging
import joblib
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .security_utils import encrypt,decrypt

#setup logging
logger=logging.getLogger("django")
# Load ML models
kmeans = joblib.load(r"D:\COLLEGE Dharmika\sldp\kmeansmodel.pkl")
regressor = joblib.load(r"D:\COLLEGE Dharmika\sldp\linearregression1.pkl")

@csrf_exempt
def predict_persona(request):
    if request.method == "POST":
        data = json.loads(request.body)
        age = data.get("age")
        income = data.get("income")
        spending_limit = data.get("spending_limit")

        if age is None or income is None or spending_limit is None:
            return JsonResponse({"error": "Missing input values"}, status=400)

        input_data = np.array([[age, income, spending_limit]])
        persona_cluster = kmeans.predict(input_data)[0]

        response = {"persona_cluster": int(persona_cluster)}
        logger.info(f"Persona Prediction: {response}")

        return JsonResponse({"persona_cluster": int(persona_cluster)})

@csrf_exempt
def predict_spending(request):
    if request.method == "POST":
        data = json.loads(request.body)
        age = data.get("age")
        income = data.get("income")

        if age is None or income is None:
            return JsonResponse({"error": "Missing input values"}, status=400)

        input_data = np.array([[age, income]])
        predicted_spending = regressor.predict(input_data)[0]

        encrypted_spending = encrypt(str(predicted_spending))

        response = {"encrypted_spending_limit": encrypted_spending}
        logger.info(f"Spending Prediction: {response}")

        return JsonResponse({"predicted_spending_limit": float(predicted_spending)})
