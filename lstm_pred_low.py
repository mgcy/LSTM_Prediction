import os
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import random
import datetime as dt

# monitor start time
start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))

# load the data and train the LSTM iteratively, deal with lower part
look_back = 2
test_index = 171

# get the number of files in upper folder
path, dirs, files = next(os.walk("D:\Academic\lstm_prediction\overlap_data\\lower"))
up_file_count = len(files)
up_test_size = int(up_file_count / 3)
up_train_size = up_file_count - up_test_size
# initialize the train and test data
trainX = numpy.zeros([1, look_back])
trainY = numpy.zeros(1)
testX = numpy.zeros([1, look_back])
testY = numpy.zeros(1)


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# get the indices of test data
test_index_list = random.sample(range(1, up_file_count), up_test_size)

for i1 in range(1, up_file_count + 1):
    # get the dir of data
    up_dir = 'D:\Academic\lstm_prediction\overlap_data\\lower\\' + str(i1) + '.txt'
    print(up_dir)
    # load the dataset
    dataset = read_csv(up_dir, header=None)
    dataset = dataset.values
    dataset = dataset.astype('float32')

    tmpX, tmpY = create_dataset(dataset=dataset, look_back=look_back)
    # show a sample such as 175.txt for plot
    if i1 == test_index:
        if i1 in test_index_list:
            show_start_index = len(testX)
        else:
            show_start_index = len(trainX)

    if i1 in test_index_list:
        testX = numpy.concatenate((testX, tmpX))
        testY = numpy.concatenate((testY, tmpY))
    else:
        trainX = numpy.concatenate((trainX, tmpX))
        trainY = numpy.concatenate((trainY, tmpY))

    if i1 == test_index:
        if i1 in test_index_list:
            show_end_index = len(testX)
        else:
            show_end_index = len(trainX)

# delete the first zeros when initialing
trainX = numpy.delete(trainX, 0, 0)
trainY = numpy.delete(trainY, 0, 0)
testX = numpy.delete(testX, 0, 0)
testY = numpy.delete(testY, 0, 0)

# combine trainX and trainY together
trainY = numpy.reshape(trainY, (trainX.shape[0], 1))
up_train_full = numpy.concatenate((trainX, trainY), axis=1)

# combine testX and testY together
testY = numpy.reshape(testY, (testX.shape[0], 1))
up_test_full = numpy.concatenate((testX, testY), axis=1)

up_full = numpy.concatenate((up_train_full, up_test_full), axis=0)

# final data structure:
#    trainX | trainY
#    ---------------
#    testX  | test Y

# normalize the whole dataset
scaler = MinMaxScaler(feature_range=(0, 1))
up_full = scaler.fit_transform(up_full)

# tmp1: whole train data
# tmp2: whole test data
tmp1 = up_full[0:len(trainX), :]
tmp2 = up_full[len(trainX):, :]

trainX = tmp1[:, 0:look_back]
trainY = tmp1[:, look_back]

testX = tmp2[:, 0:look_back]
testY = tmp2[:, look_back]

if test_index in test_index_list:
    showX = testX[show_start_index - 1:show_end_index - 1, :]
    showY = testY[show_start_index - 1:show_end_index - 1]
    print('Shown data in the test set')
else:
    showX = trainX[show_start_index - 1:show_end_index - 1, :]
    showY = trainY[show_start_index - 1:show_end_index - 1]
    print('Shown data in the train set')

# revert data
trainX = trainX[::-1]
trainY = trainY[::-1]
testX = testX[::-1]
testY = testY[::-1]
showX = showX[::-1]
showY = showY[::-1]

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
showX = numpy.reshape(showX, (showX.shape[0], 1, showX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
showPredict = model.predict(showX)

# revert Y data and predict data back
trainY = trainY[::-1]
testY = testY[::-1]
showY = showY[::-1]

trainPredict = trainPredict[::-1]
testPredict = testPredict[::-1]
showPredict = showPredict[::-1]

# invert predictions, needs change for different look_back
# for train data
tmp3 = numpy.concatenate((trainPredict, trainPredict, trainPredict), axis=1)
trainPredict = scaler.inverse_transform(tmp3)
trainPredict = trainPredict[:, 0]

tmp4 = numpy.concatenate(([trainY], [trainY], [trainY]), axis=0)
tmp4 = tmp4.transpose()
trainY = scaler.inverse_transform(tmp4)
trainY = trainY[:, 0]

# for test data
tmp5 = numpy.concatenate((testPredict, testPredict, testPredict), axis=1)
testPredict = scaler.inverse_transform(tmp5)
testPredict = testPredict[:, 0]

tmp6 = numpy.concatenate(([testY], [testY], [testY]), axis=0)
tmp6 = tmp6.transpose()
testY = scaler.inverse_transform(tmp6)
testY = testY[:, 0]

# for show data
tmp7 = numpy.concatenate((showPredict, showPredict, showPredict), axis=1)
showPredict = scaler.inverse_transform(tmp7)
showPredict = showPredict[:, 0]

tmp8 = numpy.concatenate(([showY], [showY], [showY]), axis=0)
tmp8 = tmp8.transpose()
showY = scaler.inverse_transform(tmp8)
showY = showY[:, 0]

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

# monitor end time
end_time = dt.datetime.now()
print('Stop learning {}'.format(str(end_time)))
elapsed_time = end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))

# plot show data
plt.plot(showY, label='Ori')
plt.plot(showPredict, label='Pred')
plt.show()
# plt.xlim(-1, 30)
# plt.ylim(-30, 43)
