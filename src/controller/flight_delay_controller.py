import streamlit as st
from src.view.flight_delay_view import FlightDelayView
from src.model.flight_delay_model import FlightDelayModel
import numpy as np
import pandas as pd
from scipy import stats
import time

# Controller        
class FlightDelayController:
    """
    The Controller component for the Flight Delay Prediction App.

    This class orchestrates the interaction between the Model and View components.
    """
    
    
    def __init__(self):
        self.model = FlightDelayModel()
        self.view = FlightDelayView()
        self.selected_data=self.model.selected_data()
        self.categorical_options=self.model.categorical_features()

    def run_prediction(self):
        """
        Main controller method for running the application.
        """
        self.view.display_input_form()
        self.get_user_inputs()
        input_data=self.view.display_selected_inputs(self.selected_data)
        if st.button("Predict Flight Delay"):
            flight_delay = self.model.predict_delay(input_data)
            self.view.display_predicted_delay(flight_delay)

    def get_user_inputs(self):
        """
        Collects user inputs from the Streamlit sidebar.
        """
        st.sidebar.write("Input Values:")
        # Collect input values for numerical columns
        self.selected_data['Month'] = st.sidebar.slider("Month", 1, 12)
        self.selected_data['DayofMonth'] = st.sidebar.slider("Day of Month", 1, 31)
        self.selected_data['DayOfWeek'] = st.sidebar.slider("Day of Week", 1, 7)
        
        def custom_time_input(label, default_time, key_prefix=""):
            """
            Create a custom time input widget.

            Args:
                label (str): The label for the time input.
                default_time (datetime.time): The default time value.
                key_prefix (str): A prefix for generating a unique key for the widget.

            Returns:
                datetime.time: The selected time.
            """
            st.sidebar.write(label)
            hour = st.sidebar.number_input(f"{key_prefix} Hour", min_value=0, max_value=23, value=default_time.hour)
            minute = st.sidebar.number_input(f"{key_prefix} Minute", min_value=0, max_value=59, value=default_time.minute)
            return pd.to_datetime(f"{hour:02d}:{minute:02d}").time()

        # Collect time inputs
        departure_time = custom_time_input("Departure Time", pd.to_datetime("08:00").time(), key_prefix="Dep")
        scheduled_departure_time = custom_time_input("Scheduled Departure Time", pd.to_datetime("08:00").time(), key_prefix="CRSDep")
        scheduled_arrival_time = custom_time_input("Scheduled Arrival Time", pd.to_datetime("08:00").time(), key_prefix="CRSArr")

        # Convert and store time values in dataset format
        self.selected_data['DepTime'] = int(departure_time.strftime("%H%M"))
        self.selected_data['CRSDepTime'] = int(scheduled_departure_time.strftime("%H%M"))
        self.selected_data['CRSArrTime'] = int(scheduled_arrival_time.strftime("%H%M"))

        # Other numerical inputs
        self.selected_data['FlightNum'] = st.sidebar.number_input("Flight Number", 1)
        self.selected_data['CRSElapsedTime'] = st.sidebar.number_input("CRS Elapsed Time", value=120)
        self.selected_data['AirTime'] = st.sidebar.number_input("Air Time", value=100)
        self.selected_data['DepDelay'] = st.sidebar.number_input("Departure Delay", value=0)
        self.selected_data['Distance'] = st.sidebar.number_input("Distance", value=1000)
        self.selected_data['TaxiIn'] = st.sidebar.number_input("Taxi In Time", value=10)
        self.selected_data['TaxiOut'] = st.sidebar.number_input("Taxi Out Time", value=20)
        self.selected_data['CarrierDelay'] = st.sidebar.number_input("Carrier Delay in minutes", value=0)
        self.selected_data['WeatherDelay'] = st.sidebar.number_input("Weather Delay in minutes", value=0)
        self.selected_data['NASDelay'] = st.sidebar.number_input("NAS Delay in minutes", value=0)
        self.selected_data['SecurityDelay'] = st.sidebar.number_input("Security Delay in minutes", value=0)
        self.selected_data['LateAircraftDelay'] = st.sidebar.number_input("Late Aircraft Delay in minutes", value=0)
        
        # Create select boxes for categorical features
        for feature,options in self.categorical_options.items():
            selected_value=st.sidebar.selectbox(f"Select {feature}:",options)
            self.selected_data[feature+"_"+selected_value] = 1

    
    def run_monitoring(self):
        st.title("Data & Model Monitoring App")
        st.write("You are in the Data & Model Monitoring App. Select the Date and month range from the sidebar and click 'Submit' to start model training and monitoring.")

        # Allow the user to choose their preferred date range
        new_start_month = st.sidebar.selectbox("Start Month", range(1, 7), 1)
        new_end_month = st.sidebar.selectbox("End Month", range(1, 7), 1)
        new_start_day = st.sidebar.selectbox("Start Day", range(1, 32), 1)
        new_end_day = st.sidebar.selectbox("End Day", range(1, 32), 30)
        
        
        # Select which reports to generate
        st.subheader("Select Reports to Generate")
        generate_model_report = st.checkbox("Generate Model Performance Report")
        generate_target_drift = st.checkbox("Generate Target Drift Report")
        generate_data_drift = st.checkbox("Generate Data Drift Report")
        generate_data_quality = st.checkbox("Generate Data Quality Report")

        if st.button("Submit"):
            st.write("Fetching your current batch data...")
            data_start=time.time()
            df=pd.read_csv("data/Monitoring_data.csv")
            data_end=time.time()
            time_taken=data_end - data_start
            st.write(f"Fetched the data within {time_taken:.2f} seconds")
            
            date_range = (
                    (df['Month'] >= new_start_month) & (df['DayofMonth'] >= new_start_day) &
                    (df['Month'] <= new_end_month) & (df['DayofMonth'] <= new_end_day)
                )
            # Filter data based on user input
            reference_data = df[~date_range]
            current_data = df[date_range]

            self.view.display_monitoring(reference_data,current_data)
            self.model.train_model(reference_data,current_data)

            # Generate selected reports and display them
            if generate_model_report:
                st.write("### Model Performance Report")
                st.write("Generating Model Performance Report...")
                performance_report=self.model.performance_report(reference_data,current_data)
                self.view.display_report(performance_report, "Model Performance Report")
                
            if generate_target_drift:
                st.write("### Target Drift Report")
                st.write("Generating Target Drift Report...")
                target_report=self.model.target_report(reference_data,current_data)
                self.view.display_report(target_report, "Target Drift Report")


            if generate_data_drift:
                st.write("### Data Drift Report")
                st.write("Generating Data Drift Report...")
                data_drift_report=self.model.data_drift_report(reference_data,current_data)
                self.view.display_report(data_drift_report, "Data Drift Report")

            if generate_data_quality:
                st.write("### Data Quality Report")
                st.write("Generating Data Quality Report...")
                data_quality_report=self.model.data_quality_report(reference_data,current_data)
                self.view.display_report(data_quality_report, "Data Quality Report")


