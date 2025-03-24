import sys
sys.stdout.reconfigure(encoding='utf-8')  # Set UTF-8 encoding

import joblib
from sklearn.linear_model import LinearRegression

# Re-train your model
model = LinearRegression()
# model.fit(X_train, y_train)  # Train with your data

# Save it again using scikit-learn 1.6.0
joblib.dump(model, r"D:\COLLEGE Dharmika\sldp\kmeans_model.pkl")

print("✅ Model re-saved successfully!")

