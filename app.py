import streamlit as st
import pandas as pd
from src.controller.flight_delay_controller import FlightDelayController
from sklearn.ensemble import RandomForestRegressor
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import TargetDriftPreset
from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset.regression_performance import RegressionPreset
import time
import numpy as np
from scipy import stats

def main():
    st.set_page_config(page_title="Flight Delay Prediction App", layout="wide")
    
    controller = FlightDelayController()

    # Create the Streamlit app
    st.sidebar.title("Flight Delay Prediction and Data & Model Monitoring App")
    choice = st.sidebar.radio("Select an option:", ("Make Predictions", "Monitor Data and Model"))

    if choice == "Make Predictions":
        controller.run()

    elif choice == "Monitor Data and Model":
        st.title("Data & Model Monitoring App")
        st.write("You are in the Data & Model Monitoring App. Select the Date and month range  and click 'Submit' to start model training and monitoring.")

        # Allow the user to choose their preferred date range
        new_start_month = st.sidebar.selectbox("Start Month", range(1, 7), 1)
        new_end_month = st.sidebar.selectbox("End Month", range(1, 7), 1)
        new_start_day = st.sidebar.selectbox("Start Day", range(1, 32), 1)
        new_end_day = st.sidebar.selectbox("End Day", range(1, 32), 30)
        
        st.write("Fetching the data from cloud, please wait for 1-2 minutes")
        data_start=time.time()
        df=pd.read_csv("https://flightdelay.blob.core.windows.net/flight-delayed-dataset/DelayedFlights.csv")
        data_end=time.time()
        time_taken=data_end - data_start
        st.write(f"Fetched the data from cloud within {time_taken:.2f} seconds")
        df=df.drop(['Unnamed: 0','Year','CancellationCode','TailNum','Diverted','Cancelled','ArrTime','ActualElapsedTime'],axis=1)
        delay_colns=['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

        # Impute missing values with the 0 for these columns
        df[delay_colns]=df[delay_colns].fillna(0)

        # Impute missing values with the median for these columns
        columns_to_impute = ['AirTime', 'ArrDelay', 'TaxiIn','CRSElapsedTime']
        df[columns_to_impute]=df[columns_to_impute].fillna(df[columns_to_impute].median())

        #df=pd.get_dummies(df,columns=['UniqueCarrier', 'Origin', 'Dest'], drop_first=True)

        numerical_columns = [
                'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime','CRSArrTime',
                'FlightNum', 'CRSElapsedTime', 'AirTime', 'DepDelay',
                'Distance', 'TaxiIn', 'TaxiOut', 'CarrierDelay', 'WeatherDelay', 'NASDelay',
                'SecurityDelay', 'LateAircraftDelay']

        z_threshold=3
        z_scores=np.abs(stats.zscore(df[numerical_columns]))
        outliers=np.where(z_scores>z_threshold)
        df=df[(z_scores<=z_threshold).all(axis=1)]
        date_range = (
                (df['Month'] >= new_start_month) & (df['DayofMonth'] >= new_start_day) &
                (df['Month'] <= new_end_month) & (df['DayofMonth'] <= new_end_day)
            )
        # Filter data based on user input
        reference_data = df[~date_range]
        current_data = df[date_range]

        # Select which reports to generate
        st.subheader("Select Reports to Generate")
        generate_model_report = st.checkbox("Generate Model Performance Report")
        generate_target_drift = st.checkbox("Generate Target Drift Report")
        generate_data_drift = st.checkbox("Generate Data Drift Report")
        generate_data_quality = st.checkbox("Generate Data Quality Report")

        if st.button("Submit"):
            st.write("Reference Dataset Shape:", reference_data.shape)
            st.write("Current Dataset Shape:", current_data.shape)

            # Model training
            st.write("### Model is training...")

            target = 'ArrDelay'
            numerical_features = [
                'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime', 'CRSArrTime', 'FlightNum',
                'CRSElapsedTime', 'AirTime', 'DepDelay', 'Distance', 'TaxiIn', 'TaxiOut',
                'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay'
            ]

            # Create and train the Random Forest Regressor
            rf_regressor = RandomForestRegressor(n_estimators=1, random_state=0)
            model_training_start_time = time.time()
            rf_regressor.fit(reference_data[numerical_features], reference_data[target])
            ref_prediction = rf_regressor.predict(reference_data[numerical_features])
            current_prediction = rf_regressor.predict(current_data[numerical_features])
            model_training_end_time = time.time()
            st.write(f"Time taken for Model Training: {model_training_end_time - model_training_start_time} seconds")
            
            reference_data['prediction'] = ref_prediction
            current_data['prediction'] = current_prediction
            

            # Define column mapping
            column_mapping = ColumnMapping()
            column_mapping.target = target
            column_mapping.prediction = 'prediction'
            column_mapping.numerical_features = numerical_features



            # Generate selected reports and display them
            if generate_model_report:
                st.write("### Model Monitoring")
                st.write("### Model Performance Report")
                st.write("Generated Model Performance Report")

                # Model performance report
                regression_performance_report = Report(metrics=[RegressionPreset()])
                regression_performance_report.run(
                    reference_data=reference_data,
                    current_data=current_data,
                    column_mapping=column_mapping
                )
                st.components.v1.html(regression_performance_report.get_html(), height=1000, scrolling=True)
                
            if generate_target_drift:
                st.write("### Target Drift Report")
                target_drift_report = Report(metrics=[TargetDriftPreset()])
                target_drift_report.run(
                    reference_data=reference_data,
                    current_data=current_data,
                    column_mapping=column_mapping
                )
                st.write("Generated Target Drift Report")
                st.components.v1.html(target_drift_report.get_html(), height=1000, scrolling=True)

            if generate_data_drift:
                st.write("### Data Drift Report")
                data_drift_report = Report(metrics=[DataDriftPreset()])
                data_drift_report.run(
                    reference_data=reference_data,
                    current_data=current_data,
                    column_mapping=column_mapping
                )
                st.write("Generated Data Drift Report")
                st.components.v1.html(data_drift_report.get_html(), height=1000, scrolling=True)

            if generate_data_quality:
                st.write("### Data Quality Report")
                data_quality_report = Report(metrics=[DataQualityPreset()])
                data_quality_report.run(
                    reference_data=reference_data,
                    current_data=current_data,
                    column_mapping=column_mapping
                )
                st.write("Generated Data Quality Report")
                st.components.v1.html(data_quality_report.get_html(), height=1000, scrolling=True)


if __name__ == '__main__':
    main()
