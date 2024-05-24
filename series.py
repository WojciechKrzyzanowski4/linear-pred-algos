import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# the data is temporary

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

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


def predict_result(var1, var2, date_str, df, model):
    date = datetime.strptime(date_str, '%d.%m.%Y')
    date_num = (date - df['date'].min()).days
    prev_date = date - pd.Timedelta(days=1)


    if not df[df['date'] == prev_date].empty:
        var1_lag1 = df[df['date'] == prev_date]['var1'].values[0]
        var2_lag1 = df[df['date'] == prev_date]['var2'].values[0]

        
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








