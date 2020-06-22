import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM

from yahoo_fin import stock_info as si
import datetime


def new_dataset(dataset, step_size):
    data_X, data_Y = [], []
    for i in range(len(dataset) - step_size - 1):
        a = dataset[i:(i + step_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + step_size, 0])
    return np.array(data_X), np.array(data_Y)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

np.random.seed(7)

stocks = ['AMD', 'AMZN', 'AAPL', 'GOOG', 'MSFT', 'NFLX', 'TSLA', 'INR=X']
stock_dic = {"AMD": "AMD Inc.", "AMZN": "Amazon.com, Inc. ", "AAPL": "Apple Inc.", "GOOG": "Alphabet Inc.", "MSFT": "Microsoft Corporation" , "NFLX": "Netflix Inc." ,"TSLA": "Tesla Inc.", "INR=X": "USD/INR"}
print('Index Ticker Company')
for i in range(len(stocks)):
    print(i+1,stocks[i],stock_dic[stocks[i]])
index = int(input("Enter the index "))-1
stock = stocks[index]
stock_name = stock_dic[stock]

datelive = datetime.date.today().strftime("%d/%m/%Y")
datalive = si.get_data(stock, start_date='01/06/2008', end_date=datelive, index_as_date=False, interval="1d")
dataset = datalive.loc[:, ['open', 'high', 'low', 'adjclose']]

obs = np.arange(1, len(dataset) + 1, 1)

OHLC_avg = dataset.mean(axis=1)
HLC_avg = dataset[['high', 'low', 'adjclose']].mean(axis=1)
close_val = dataset[['adjclose']]

plt.title(stock_name + " Stocks")
plt.plot(obs, OHLC_avg, 'r', label='OHLC avg')
plt.plot(obs, HLC_avg, 'b', label='HLC avg')
plt.plot(obs, close_val, 'g', label='Closing price')
plt.legend(loc='upper right')
plt.show()

OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg), 1))  # 1664
scaler = MinMaxScaler(feature_range=(0, 1))
OHLC_avg = scaler.fit_transform(OHLC_avg)

train_OHLC = int(len(OHLC_avg) * 0.4)
test_OHLC = len(OHLC_avg) - train_OHLC
train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC, :], OHLC_avg[train_OHLC:len(OHLC_avg), :]

trainX, trainY = new_dataset(train_OHLC, 1)
testX, testY = new_dataset(test_OHLC, 1)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
step_size = 1

model = Sequential()
model.add(LSTM(32, input_shape=(1, step_size), return_sequences=True))
model.add(LSTM(16))
model.add(Dense(1))
model.add(Activation('linear'))

model.compile(loss='mean_squared_error', optimizer='adagrad')
model.fit(trainX, trainY, epochs=2000, batch_size=256, verbose=2)
model.save('mymodel.hdf5')

loadedModel = keras.models.load_model('mymodel.hdf5')
trainPredict = loadedModel.predict(trainX)
testPredict = loadedModel.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train RMSE: %.2f' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test RMSE: %.2f' % (testScore))

trainPredictPlot = np.empty_like(OHLC_avg)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[step_size:len(trainPredict) + step_size, :] = trainPredict

testPredictPlot = np.empty_like(OHLC_avg)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (step_size * 2) + 1:len(OHLC_avg) - 1, :] = testPredict

OHLC_avg = scaler.inverse_transform(OHLC_avg)

plt.title(stock_name + " Stocks prediction")
plt.plot(OHLC_avg, 'g', label='original dataset')
plt.plot(trainPredictPlot, 'r', label='training set')
plt.plot(testPredictPlot, 'b', label='predicted stock price/test set')
plt.legend(loc='upper right')
plt.xlabel('Time in Days')
plt.ylabel('OHLC Value of ' + stock_name + ' Stocks')
plt.show()

last_val = testPredict[-1]
last_val_scaled = last_val / last_val
next_val = loadedModel.predict(np.reshape(last_val_scaled, (1, 1, 1)))
print("Last Day Value:", np.ndarray.item(last_val))
print("Next Day Value:", np.ndarray.item(last_val * next_val))
