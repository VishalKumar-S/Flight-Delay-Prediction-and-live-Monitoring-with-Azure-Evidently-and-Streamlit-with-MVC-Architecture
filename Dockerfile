# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Create the necessary directories
RUN mkdir -p /app/data /app/models /app/src/controller /app/src/model /app/src/view

# Copy the files from your host machine into the container
COPY app.py /app/
COPY src/controller/flight_delay_controller.py /app/src/controller/
COPY src/model/flight_delay_model.py /app/src/model/
COPY src/view/flight_delay_view.py /app/src/view/
COPY requirements.txt /app/

# Create and activate a virtual environment
RUN python -m venv venv
RUN /bin/bash -c "source venv/bin/activate"

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Install wget
RUN apt-get update && apt-get install -y wget

# Download the dataset file using wget
RUN wget -O /app/data/DelayedFlights.csv https://flightdelay.blob.core.windows.net/flight-delayed-dataset/DelayedFlights.csv

# Download the best model file using wget
RUN wget -O /app/models/best_model.pkl https://flightdelay.blob.core.windows.net/flight-delayed-dataset/best_model.pkl

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define the default command to run your Streamlit application
CMD ["streamlit", "run", "app.py"]
