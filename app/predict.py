from app.models import load_stock_model
# from app.predict import predict_stock_prices
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import load_model



def predict_stock_prices_route(ticker_symbol):
    model = load_stock_model("AAPL")
    next_month_dates, predicted_prices, _ = predict_stock_prices(ticker_symbol, model)

    img = BytesIO()
    plt.figure(figsize=(10, 6))
    plt.plot(next_month_dates, predicted_prices, label='Predicted Close Price', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(f'Stock Price Prediction for {ticker_symbol} Next Month')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return plot_url


def predict_stock_prices(ticker_symbol, model):
    # Fetch historical stock prices from Yahoo Finance for the last month
    end_date = pd.Timestamp.now()-pd.DateOffset(months=1)
    start_date = end_date - pd.DateOffset(months=2)
    historical_data = yf.download(ticker_symbol, start=start_date, end=end_date)

    # Extract the 'Close' prices from the historical data
    close_prices = historical_data['Close']

    # Normalize the data using Min-Max scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices_scaled = scaler.fit_transform(close_prices.values.reshape(-1, 1))

    # Define sequence length for LSTM model
    seq_length = 10

    # Create sequences of data for LSTM model
    def create_sequences(data, seq_length):
        X = []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length), 0])
        return np.array(X)

    # Create sequences of data for LSTM model
    X = create_sequences(close_prices_scaled, seq_length)

    # Reshape input data to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Make predictions using the LSTM model
    predicted_scaled = model.predict(X)

    # Invert the predictions to the original scale
    predicted = scaler.inverse_transform(predicted_scaled)

    # Invert actual values to original scale for comparison
    actual_values = scaler.inverse_transform(close_prices_scaled[seq_length:])

    # Calculate R-squared (R^2) Score
    r2 = r2_score(actual_values, predicted)

    # Generate dates for the next month
    next_month_dates = pd.date_range(end_date + pd.DateOffset(days=1), periods=len(predicted))

    return next_month_dates, predicted, actual_values
