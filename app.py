from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import load_model
import matplotlib

from app.models import load_stock_model
matplotlib.use('Agg')  # Use the 'Agg' backend which is non-interactive
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
from app.models import load_stock_model
#from app.predict import predict_stock_prices
from app.mvplot import plot_stock_data
from app.predict import predict_stock_prices_route,predict_stock_prices


app = Flask(__name__)



@app.route('/trendline', methods=['POST'])
def trendline():
    if request.method == 'POST':
        stock = request.form['stock']
        start = request.form['start']
        end = request.form['end']
        plot_url = plot_stock_data(stock, start, end)
        return render_template('result.html', plot_url=plot_url)
    else:
        return render_template('index.html')
    
#model = load_model('models/AMZN_model.keras')

@app.route('/performance',methods=['POST', 'GET'])
def performance():
    if request.method == 'POST':
        ticker_symbol = request.form['ticker_symbol']
        model = load_stock_model(ticker_symbol)

        # Predict stock prices
        _ ,test_predictions,actual_values = predict_stock_prices(ticker_symbol, model)

        # Visualize the predictions and actual values
        plt.figure(figsize=(10, 6))
        plt.plot(test_predictions, label='Predictions')
        plt.plot(actual_values, label='Actual Data')
        plt.xlabel('Time')
        plt.ylabel('Close Price')
        plt.title(f'LSTM Stock Price Prediction for {ticker_symbol} Testing Data')
        plt.legend()

        # Save the plot to a bytes buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Encode the bytes buffer to base64
        plot_data = base64.b64encode(buffer.read()).decode('utf-8')

        # Render the template with plot data and R-squared score
        return render_template('result.html', plot_data=plot_data)
    else:
        return render_template('index.html')


# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for predicting stock prices

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        ticker_symbol = request.form['ticker_symbol']
        plot_url = predict_stock_prices_route(ticker_symbol)
        return render_template('result.html', plot_url=plot_url)
    else:
        return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)
