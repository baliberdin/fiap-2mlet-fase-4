import yfinance as yf
import numpy as np

def retrieve_stock_data(ticker, start_dt):
    data = yf.download(ticker, start=start_dt)
    # Deixa apenas o preço de fechamento
    data.drop(columns=['Volume','High', 'Low', 'Open'], inplace=True)
    return data  

def format_timeseries_dataset(data, lookback=60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])  # Últimos N registros
        y.append(data[i, 0])  # Próximo valor
    X, y = np.array(X), np.array(y)
    return X, y