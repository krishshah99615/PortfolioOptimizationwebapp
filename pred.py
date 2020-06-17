from pandas_datareader import data
import pandas as pd
import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import glob


def get_data(name, t='daily'):

    # Get Data from past 12 years from the current day
    start_date = datetime.datetime.now() + relativedelta(years=-11)
    end_date = datetime.datetime.now() + relativedelta(years=-1)

    # Get Daily data with holidays
    df = data.DataReader(name, 'yahoo', start_date, end_date)
    # Pad holidays with previous value
    df = df.asfreq('D', method='pad')

    # Convert Type of data[by mean]
    if t == 'yearly':
        df = df.resample('Y').mean()
        return df
    if t == 'monthly':
        df = df.resample('M').mean()
        return df
    else:
        return df


def create_dataset(d):

    t = d['Open']
    t = pd.DataFrame(t)
    sc = MinMaxScaler(feature_range=(0, 1))
    ts = sc.fit_transform(t)

    X_train = []
    y_train = []
    for i in range(60, len(ts)):
        X_train.append(ts[i-60:i, 0])
        y_train.append(ts[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return X_train, y_train, sc


def get_model(X_train):
    m = Sequential()
    m.add(LSTM(units=50, return_sequences=True,
               input_shape=(X_train.shape[1], 1)))
    m.add(Dropout(0.2))
    m.add(LSTM(units=50, return_sequences=True))
    m.add(Dropout(0.2))
    m.add(LSTM(units=50, return_sequences=True))
    m.add(Dropout(0.2))
    m.add(LSTM(units=50))
    m.add(Dropout(0.2))
    m.add(Dense(units=1))

    return m


def train(m, X_train, y_train, e):

    m.compile(optimizer='adam', loss='mean_squared_error')
    history = m.fit(X_train, y_train, epochs=e, batch_size=32)

    return m


def get_test_data(name, orignal_data, sc):

    start_date = datetime.datetime.now() + relativedelta(years=-1)
    end_date = datetime.datetime.now()

    df = data.DataReader(name, 'yahoo', start_date, end_date)

    test_set = df['Open']
    test_set = pd.DataFrame(test_set)

    dataset_total = pd.concat((orignal_data['Open'], df['Open']), axis=0)

    inputs = dataset_total[len(dataset_total) - len(df) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(60, 60+len(df)):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_test, test_set


def predict_on_test(model, X_test, real_val, sc, name):

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    predicted_stock_price = pd.DataFrame(
        predicted_stock_price, index=real_val.index)

    '''plt.plot(real_val, color='red', label=f'Real {name} Stock Price')
    plt.plot(predicted_stock_price, color='blue',
             label=f'Predicted {name} Stock Price')
    plt.title(f'{name} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'{name} Stock Price')
    plt.legend()'''

    return predicted_stock_price, real_val


def get_test_graph(choice, m):

    dataset = get_data(choice)
    X_train, y_train, sc = create_dataset(dataset)

    X_test, real_val = get_test_data(choice, dataset, sc)
    predicted_stock_price, real_val = predict_on_test(
        m, X_test, real_val, sc, choice)

    return predicted_stock_price, real_val, sc


def pred_next(model, sc, l=5):
    seed = data.DataReader('GOOGL', 'yahoo', datetime.datetime.today(
    )+relativedelta(days=-100), datetime.datetime.today())
    seed = seed.asfreq('D', method='pad')
    seed = seed[-60:]['Open'].values
    seed = seed.reshape(-1, 1)
    seed = sc.transform(seed).reshape(1, -1, 1)
    # p=sc.inverse_transform(model.predict(seed))

    p = model.predict(seed)
    preds = [p[0][0]]
    new_seed = seed[0][1:]
    for i in range(l):

        new_seed = np.append(new_seed, preds[i])
        a = new_seed.reshape(1, -1, 1)
        preds.append((model.predict(a))[0][0])
        new_seed = new_seed[1:]
    preds = sc.inverse_transform(np.array(preds).reshape(-1, 1))
    preds = pd.DataFrame(preds, index=pd.date_range(start=datetime.datetime.today().strftime(
        '%Y-%m-%d'), end=(datetime.datetime.today()+relativedelta(days=l)).strftime('%Y-%m-%d')))
    return preds

    # return p[0][0]


# predicted_stock_price, real_val, sc = get_test_graph('GOOGL')
# pred_next(tf.keras.models.load_model(glob.glob('*GOOGL.h5')[0]), sc)
