import pandas as pd
import numpy as np
import datetime as datetime
from dateutil import parser
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout

data = {
    'var1': [i for i in range(1, 50)],
    'var2': [i*2 for i in range(1, 50)],
    'result': [i*3 for i in range(1, 50)],
    'date': pd.date_range(start='01.01.1900', periods=49, freq='D')
}

df = pd.DataFrame(data)

df['date_num'] = (df['date'] - df['date'].min()).dt.days

df['var1_lag1'] = df['var1'].shift(1)
df['var2_lag1'] = df['var2'].shift(1)

df.dropna(inplace=True)

X = df[['var1', 'var2', 'date_num', 'var1_lag1', 'var2_lag1']]
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.1),  # Dropout layer to prevent overfitting
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


def predict_result(var1, var2, date_str, df, model):
    try:
        date = parser.parse(date_str)
        print(date)
    except ValueError:
        print("Error: Date format not recognized.")
        return None

    date_num = (date - df['date'].min()).days
    prev_date = date - pd.Timedelta(days=1)

    # Sprawdzenie, czy poprzednia data istnieje w DataFrame
    if not df[df['date'] == prev_date].empty:
        var1_lag1 = df[df['date'] == prev_date]['var1'].values[0]
        var2_lag1 = df[df['date'] == prev_date]['var2'].values[0]

        # Przygotowanie cech w formie DataFrame z odpowiednimi nazwami kolumn
        features = pd.DataFrame({
            'var1': [var1],
            'var2': [var2],
            'date_num': [date_num],
            'var1_lag1': [var1_lag1],
            'var2_lag1': [var2_lag1]
        })

        return model.predict(features)[0]
    else:
        print("Opóźnione wartości dla podanej daty nie są dostępne.")
        return None

print(predict_result(50, 100, '25.01.1900', df, model))