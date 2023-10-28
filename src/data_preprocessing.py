from azure.storage.blob import BlobServiceClient
import io
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np


class DataPreprocessorTemplate:
    """
    Template method pattern for data preprocessing with customizable steps.
    """
    def __init__(self, data_url):
        """
        Initialize the DataPreprocessor Template with the data URL including the SAS token.

        Args:
            data_url (str): The URL to the data with the SAS token.
        """
        self.data_url = data_url
        self.numerical_columns = [
        'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime','CRSArrTime',
        'FlightNum', 'CRSElapsedTime', 'AirTime', 'DepDelay',
        'Distance', 'TaxiIn', 'TaxiOut', 'CarrierDelay', 'WeatherDelay', 'NASDelay',
        'SecurityDelay', 'LateAircraftDelay']

    def fetch_data(self):
        """
        Fetch data from Azure Blob Storage.

        Returns:
            pd.DataFrame: The fetched dataset as a Pandas DataFrame.
        """
        try:
            # Fetch the dataset using the provided URL
            print("Fetching the data from cloud...")
            data = pd.read_csv(self.data_url)
            return data
        except Exception as e:
            raise Exception("An error occurred during data retrieval: " + str(e))
        
    def clean_data(self,df):
        """
        Clean and preprocess the input data.

        Args:
            df (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The cleaned and preprocessed dataset.
        """
        print("Cleaning data...")
        df=self.remove_features(df)
        df=self.impute_missing_values(df)
        df=self.encode_categorical_features(df)
        df=self.remove_outliers(df)
        return df

    def remove_features(self,df):
        """
        Remove unnecessary columns from the dataset.

        Args:
            df (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The dataset with unnecessary columns removed.
        """
        print("Removing unnecessary columns...")
        df=df.drop(['Unnamed: 0','Year','CancellationCode','TailNum','Diverted','Cancelled','ArrTime','ActualElapsedTime'],axis=1)
        return df

    def impute_missing_values(self,df):
        """
        Impute missing values in the dataset.

        Args:
            df (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The dataset with missing values imputed.
        """
        print("Imputing missing values...")
        delay_colns=['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
        
        # Impute missing values with the 0 for these columns
        df[delay_colns]=df[delay_colns].fillna(0)

        # Impute missing values with the median for these columns
        columns_to_impute = ['AirTime', 'ArrDelay', 'TaxiIn','CRSElapsedTime']
        df[columns_to_impute]=df[columns_to_impute].fillna(df[columns_to_impute].median())
        return df

    def encode_categorical_features(self,df):
        """
        Encode categorical features in the dataset.

        Args:
            df (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The dataset with categorical features encoded.
        """
        print("Encoding categorical features...")
        df=pd.get_dummies(df,columns=['UniqueCarrier', 'Origin', 'Dest'], drop_first=True)
        return df
    
    

    def remove_outliers(self,df):
        """
        Remove outliers from the dataset.

        Args:
            df (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The dataset with outliers removed.
        """
        print("Removing outliers...")
        z_threshold=3
        z_scores=np.abs(stats.zscore(df[self.numerical_columns]))
        outliers=np.where(z_scores>z_threshold)
        df_no_outliers=df[(z_scores<=z_threshold).all(axis=1)]
        print("Shape after data cleaning:", df_no_outliers.shape)
        return df_no_outliers
    
    def save_cleaned_data(self,cleaned_data, output_path):
        """
        Save the cleaned data to a CSV file.

        Args:
            cleaned_data (pd.DataFrame): The cleaned dataset.
            output_path (str): The path to save the cleaned data.
        """
        print("Saving cleaned data...")
        cleaned_data.to_csv(output_path,index=False)

def main():
    # URL to the data including the SAS token
    data_url = "https://flightdelay.blob.core.windows.net/flight-delayed-dataset/DelayedFlights.csv"

    output_path = "../data/cleaned_flight_delays.csv"

    data_preprocessor = DataPreprocessorTemplate(data_url)
    data = data_preprocessor.fetch_data()
    cleaned_data=data_preprocessor.clean_data(data)
    data_preprocessor.save_cleaned_data(cleaned_data, output_path)

if __name__ == "__main__":
    main()