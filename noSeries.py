import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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


# we prepare the input stuff as well as our result stuff and we happy

X = df[['var1', 'var2', 'date_num']]
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

    features = pd.DataFrame({
        'var1': [var1],
        'var2': [var2],
        'date_num': [date_num],
    })

    return model.predict(features)[0]


# lets hope something happens and we see 150

print(predict_result(50, 100, '27.02.1900', df, model))