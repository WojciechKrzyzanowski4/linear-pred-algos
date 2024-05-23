import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# the data is temporary but the import thing is that we will
# have the date as the main thing that we take into consideration

data = {
    'var1': [i for i in range(1, 50)],
    'var2': [i*2 for i in range(1, 50)],
    'result': [i*3 for i in range(1, 50)],
    'date': pd.date_range(start='01.01.1900', periods=49, freq='D')
}

# creating the dataframe

df = pd.DataFrame(data)

# we will convert the dates in to their offset from the
# first date and set the amount of days passed since then

df['date_num'] = (df['date'] - df['date'].min()).dt.days

# we need the lagging variables to implement "SZEREG CZASOWY"

df['var1_lag1'] = df['var1'].shift(1)
df['var2_lag1'] = df['var2'].shift(1)

# make sure to remove the nan

df.dropna(inplace=True)

# we prepare the input stuff as well as our result stuff and we happy

X = df[['var1', 'var2', 'date_num', 'var1_lag1', 'var2_lag1']]
y = df['result']

# Classic 80 20 split to train the model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Let's train our model

model = LinearRegression()
model.fit(X_train, y_train)

# we shall predict the results

y_pred = model.predict(X_test)

# Let's check how accurate we were

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


def predict_result(var1, var2, date_str, df, model):
    date = datetime.strptime(date_str, '%d.%m.%Y')
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


# lets hope to good something happens and we see 150

print(predict_result(50, 100, '25.01.1900', df, model))








