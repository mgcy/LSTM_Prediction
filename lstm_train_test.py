import pandas
import matplotlib.pyplot as plt
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import datetime as dt


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def lstm_predtiction(datadir, look_back, test_size):
    # load the upper part data
    dataset = pandas.read_csv(datadir, header=None)
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train, test = dataset[0:len(dataset) - test_size, :], dataset[len(dataset) - test_size:len(dataset), :]
    print('The lengths of training data and test data are:')
    print(len(train), len(test))

    # reshape into X=t and Y=t+look_back
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=500, batch_size=1, verbose=2,shuffle=False)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))

    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

    return trainScore, testScore, trainPredict, testPredict


def plot_upper(ori_dataset, trainPredict, testPredict, look_back):
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(ori_dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(ori_dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict) + (look_back * 2):len(trainPredict) + (look_back * 2) + len(testPredict),
    :] = testPredict

    return trainPredictPlot, testPredictPlot


def plot_lower(ori_dataset, trainPredict, testPredict, look_back):
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(ori_dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[-look_back - len(trainPredict):- look_back, :] = trainPredict[::-1]
    # trainPredictPlot[37:60, :] = trainPredict[::-1]

    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(ori_dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[
    -look_back - len(trainPredict) - look_back - len(testPredict):-look_back - len(trainPredict) - look_back,
    :] = testPredict[::-1]
    return trainPredictPlot, testPredictPlot


if __name__ == '__main__':
    # monitor start time
    start_time = dt.datetime.now()
    print('Start learning at {}'.format(str(start_time)))

    # set training parameters
    test_size = 11 #look_back+overlap
    look_back = 3

    # load the data
    dataset_full_dir = 'D:\Academic\lstm_prediction\\test_data.txt'
    dataset_upper_dir = 'D:\Academic\lstm_prediction\\test_data_upperpart_test.txt'
    dataset_lower_dir = 'D:\Academic\lstm_prediction\\test_data_lowerpart_test.txt'

    # fit the LSTM model
    up_train_mse, up_test_mse, up_train_prediction, up_test_prediction = lstm_predtiction(datadir=dataset_upper_dir,
                                                                                          look_back=look_back,
                                                                                          test_size=test_size)
    lower_train_mse, lower_test_mse, lower_train_prediction, lower_test_prediction = lstm_predtiction(
        datadir=dataset_lower_dir,
        look_back=look_back,
        test_size=test_size)

    ori_dataset = pandas.read_csv(dataset_full_dir, header=None)
    # ori_dataset = ori_dataset.astype('float32')

    # plot the training and prediction
    up_train_plot, up_test_plot = plot_upper(ori_dataset=ori_dataset, trainPredict=up_train_prediction,
                                             testPredict=up_test_prediction, look_back=look_back)
    lower_train_plot, lower_test_plot = plot_lower(ori_dataset=ori_dataset, trainPredict=lower_train_prediction,
                                                   testPredict=lower_test_prediction, look_back=look_back)
    # print mse of training and testing
    print('\n')
    print('Upper Train Score: %.2f RMSE' % (up_train_mse))
    print('Upper Test Score: %.2f RMSE' % (up_test_mse))
    print('Lower Train Score: %.2f RMSE' % (lower_train_mse))
    print('Lower Test Score: %.2f RMSE' % (lower_test_mse))

    # monitor end time
    end_time = dt.datetime.now()
    print('Stop learning {}'.format(str(end_time)))
    elapsed_time = end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)))

    # plot baseline and predictions
    plt.plot(ori_dataset)
    plt.plot(up_train_plot)
    plt.plot(up_test_plot)
    plt.plot(lower_train_plot)
    plt.plot(lower_test_plot)
    plt.show()
