from tensorflow.keras.models import Sequential, LSTM, Dense
from tensorflow.keras import optimizers

class StockLSTM(Sequential):
    
    def __init__(self, units: int= 50, learning_rate: float = 0.0007):
        self.add(LSTM(units=units, return_sequences=False))
        self.add(Dense(units=1))
        self.compile(optimizer=optimizers.Adam(learning_rate=learning_rate) , loss='mean_squared_error', metrics=['mae'])
