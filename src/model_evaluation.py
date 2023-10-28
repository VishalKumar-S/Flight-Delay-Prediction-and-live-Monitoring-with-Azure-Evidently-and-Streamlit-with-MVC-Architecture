# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# Load the cleaned dataset
cleaned_data = pd.read_csv("../data/cleaned_flight_delays.csv")

# Load the trained machine learning models
random_forest_model = joblib.load("../models/random_forest_model.pkl")
linear_regression_model = joblib.load("../models/linear_regression_model.pkl")
xgboost_model = joblib.load("../models/xgboost_model.pkl")
ridge_regression_model = joblib.load("../models/ridge_regression_model.pkl")
lightgbm_model = joblib.load("../models/lightgbm_model.pkl")

# Define the target variable (ArrDelay) and features (X)
target_variable = "ArrDelay"
X = cleaned_data.drop(columns=[target_variable])
y = cleaned_data[target_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2

# Create a dictionary of models to evaluate
models = {
    "Random Forest": random_forest_model,
    "Linear Regression": linear_regression_model,
    "XGBoost": xgboost_model,
    "Ridge Regression": ridge_regression_model,
    "LightGBM": lightgbm_model,
}

# Evaluate each model and store their metrics
metrics = {}
for model_name, model in models.items():
    mae, mse, r2 = evaluate_model(model, X_test, y_test)
    metrics[model_name] = {"MAE": mae, "MSE": mse, "R2": r2}

# Print the evaluation metrics for all models
for model_name, model_metrics in metrics.items():
    print(f"Metrics for {model_name}:")
    print(f"Mean Absolute Error (MAE): {model_metrics['MAE']:.2f}")
    print(f"Mean Squared Error (MSE): {model_metrics['MSE']:.2f}")
    print(f"R-squared (R2) Score: {model_metrics['R2']:.2f}")
    print()

# Find the best model based on R2 score
best_model = max(metrics, key=lambda model: metrics[model]["R2"])

# Print the results
print(f"The best model out of all the trained models is {best_model} with the following metrics:")
print(f"Mean Absolute Error (MAE): {metrics[best_model]['MAE']}")
print(f"Mean Squared Error (MSE): {metrics[best_model]['MSE']}")
print(f"R-squared (R2) Score: {metrics[best_model]['R2']}")

# Save the best model for later use
joblib.dump(models[best_model], "../models/best_model.pkl")
print("Best model saved as best_model.pkl")


