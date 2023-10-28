import pandas as pd
import joblib

# Model
class FlightDelayModel:
    """
    The Model component for the Flight Delay Prediction App.

    This class handles data loading, model loading, and delay predictions.
    """

    def __init__(self, model_file="models/best_model.pkl"):
        """
        Initializes the FlightDelayModel.

        Args:
            data_file (str): Path to the CSV file containing flight data.
            model_file (str): Path to the pre-trained machine learning model.
        """
        
        self.numerical_columns = [
        'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime', 'CRSArrTime',
        'FlightNum', 'CRSElapsedTime', 'AirTime', 'DepDelay',
        'Distance', 'TaxiIn', 'TaxiOut', 'CarrierDelay', 'WeatherDelay', 'NASDelay',
        'SecurityDelay', 'LateAircraftDelay']
    
        

        self.model = joblib.load(model_file)

    def predict_delay(self, input_data):
        """
        Predicts flight delay based on user input.

        Args:
            input_data (pd.DataFrame): User-provided input data.

        Returns:
            float: Predicted flight delay in minutes.
        """
        return self.model.predict(input_data)
