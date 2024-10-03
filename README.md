# Time Series Receipt Prediction with LSTM

This project is a Flask-based web application that predicts the number of scanned receipts for a certain number of future days based on past daily receipt counts. It uses an LSTM (Long Short-Term Memory) neural network model built with PyTorch for time series forecasting.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [How to Use](#how-to-use)

## Introduction

This project is designed to forecast the number of receipts scanned for a specified number of future days using data from the past 12 days. The predictions are displayed in both numeric format and graphical form. The model is trained using historical data and can predict any number of future days based on previous receipt counts.

## Features

- Input daily receipt counts for the past 12 days.
- Specify the number of future days to predict.
- View the predictions in both numeric format and a plotted graph.
- Utilizes LSTM, a type of recurrent neural network, for time series prediction.

## Technologies Used

- **Python**: The main programming language used for both the model and the web app.
- **Flask**: A lightweight web framework for Python to handle the web interface.
- **PyTorch**: The deep learning framework used for training the LSTM model.
- **NumPy**: Used for numerical operations and data manipulation.
- **Matplotlib**: Used for generating plots for visualizing the receipt counts and predictions.
- **HTML/CSS**: Frontend of the web app for displaying forms and results.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/timeseries-receipt-prediction.git
   cd timeseries-receipt-prediction
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure PyTorch is installed with GPU support (optional). For CUDA support, install the latest version of PyTorch with CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. Run the Flask app:
   ```bash
   python app.py
   ```

5. Open your web browser and navigate to `http://127.0.0.1:5000/`.

## How to Use

1. Enter the receipt counts for the past 12 days in the input fields.
2. Specify how many future days you want predictions for (e.g., 5, 10, 20 days).
3. Click the submit button.
4. The predicted receipt counts will be displayed, along with a line plot showing both past and predicted receipt counts.
