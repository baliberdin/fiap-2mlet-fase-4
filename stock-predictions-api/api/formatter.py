import pandas as pd
import datetime

def format_to_response_data(data: pd.DataFrame, predicted_data: pd.DataFrame, predictions, model_version, prediction_days: int, history_days: int):
    np_data = data.to_numpy()
    np_predicted_data = predicted_data.to_numpy()
    
    # Formata os dados de preços reais para serem enviados no retorno da API
    response_data = []
    for reg in np_data:
        response_data.append({"date":reg[0], "close":reg[1]})
        
    # Formata os dados de preços preditos para serem enviados no retorno da API
    predicted_response_data = []
    for reg in np_predicted_data:
        predicted_response_data.append({"date":reg[0], "close":reg[1]})

    # Trata a última data para saber se é um final de semana
    last_date = data.iloc[-1]['reference_date'] + datetime.timedelta(days=1)
    while last_date.weekday() in [5,6]:
        last_date = last_date + datetime.timedelta(days=1)

    future_prices = []
    i = 0
    for price in predictions:
        future_prices.append({"date": last_date+datetime.timedelta(days=i), "close": price})
        i = i+1
    
    return {"model": model_version, 
            "prediction_days":prediction_days,"real_price":response_data[-history_days:], 
            "predicted_price":predicted_response_data[-history_days:], 
            "future_price":future_prices[:prediction_days]}