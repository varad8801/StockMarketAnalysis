import os
from keras.models import load_model

def load_stock_model(ticker_symbol):
    model_filename = f'Saved Models/{ticker_symbol}.h5'
    if os.path.exists(model_filename):
        return load_model(model_filename)
    else:
        return load_model('Saved Models/ns10_close.keras')
