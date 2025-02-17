from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import tensorflow as tf

def create_model(units: int = 50, learning_rate: float= 0.0001, dropout_rate: float = 0.2):
    model = Sequential()
    model.add(Input((1, 1)))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate) , loss='mean_squared_error', metrics=['mae'])
    return model