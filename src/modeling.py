# modeling.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, RidgeCV, LassoCV
import xgboost as xgb
import lightgbm as lgb
import joblib
import time

# Factory Method for creating machine learning models
def create_model(model_name):
    if model_name == "random_forest":
        return RandomForestRegressor(n_estimators=3, random_state=42)
    elif model_name == "linear_regression":
        return LinearRegression()
    elif model_name == "xgboost":
        return xgb.XGBRegressor()
    elif model_name == "ridge_regression":
        return Ridge(alpha=1.0)  # Adjust alpha as needed
    elif model_name == "lightgbm":
        return lgb.LGBMRegressor()


# Load the cleaned dataset
print("Model loading started...")
cleaned_data = pd.read_csv("../data/cleaned_flight_delays.csv")
print("Model loading completed")

# Define the target variable (ArrDelay) and features (X)
target_variable = "ArrDelay"
X = cleaned_data.drop(columns=[target_variable])
y = cleaned_data[target_variable]

# Split the data into training and testing sets
print("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split completed.")

# Function to train and save a model
def train_and_save_model(model_name, X_train, y_train):
    model = create_model(model_name)

    print(f"Training the {model_name} model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    print("Model training completed...")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training time taken to complete: {elapsed_time:.2f} seconds")
    
    # Save the trained model for later use
    joblib.dump(model, f"../models/{model_name}_model.pkl")
    print(f"{model_name} model saved as {model_name}_model.pkl")

# Create and train a Random Forest model
train_and_save_model("random_forest", X_train, y_train)

# Train the linear regression model
train_and_save_model("linear_regression", X_train, y_train)

# Create and train an XGBoost model
train_and_save_model("xgboost", X_train, y_train)

# Create and train additional models
train_and_save_model("ridge_regression", X_train, y_train)

train_and_save_model("lightgbm", X_train, y_train)
