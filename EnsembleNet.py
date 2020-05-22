# -*- coding: utf-8 -*-
"""
data：2019/11/29
CNN Model
author: fengMAO
"""

import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # 数据处理
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from numpy import moveaxis
import time

# global variable
# file_name = '../../data/zone_taxi_demand/10min_interval'  # taxi demand file
# file_1 = '../../data/mutual_information/zone_correlation10.csv'
file_name = '../../data/zone_taxi_demand/15min_interval'  # taxi demand file
file_1 = '../../data/mutual_information/zone_correlation15.csv'
lags = 4  # lagged number for prediction
time_interval = 15  # time interval
pred = 30


# ——————————————— Data process ———————————————————
# traverse all files in the specified folder
def eachFile(filepath,list_name):
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        list_name.append(child)
    # list_name.sort()

# white noise processing
def noise_process(data):
    x = []  # column index
    for i in range(200):
        x.append(i)

    add_data = pd.DataFrame(columns = x)
    for i in range(4):
        noise = pd.DataFrame(np.random.randint(0, 2, (1, 200)), columns=x)
        add_data = add_data.append(noise + data, ignore_index=True)

    return add_data

# get taxi demand
def get_taxi_demand(list_name):
    # build dataframe
    x = []
    for i in range(200):
        x.append(i)
    taxi_demand = pd.DataFrame(columns = x)
    all_data = pd.DataFrame(columns = x) # all data

    # get taxi demand of each day
    for i, file in zip(range(len(list_name)), list_name):
        print(file)
        day_demand = (pd.read_csv(file)).iloc[:, 2:]

        # transpose
        day_demand = pd.DataFrame(day_demand.values.T, index = day_demand.columns, columns = day_demand.index)
        add_data = noise_process(day_demand.iloc[0,:])

        # all_data
        all_data = all_data.append(add_data,ignore_index=True)
        all_data = all_data.append(day_demand, ignore_index=True)

        if i == 0:
            taxi_demand = day_demand
        else:
            taxi_demand = taxi_demand + day_demand

    # average taxi demand
    taxi_demand = (np.array(taxi_demand))/len(list_name)

    train_data = all_data.iloc[:(len(list_name) - 1)*(N+4), ]
    test_data = all_data.iloc[(len(list_name) - 1)*(N+4):,]


    print(taxi_demand.shape)
    print(train_data)
    print(test_data.iloc[4:,:])

    return taxi_demand, all_data

# Image features building
def data_process_CNN(train_data):
    # train_data = np.array(train_data)
    print('train_data', train_data.shape)

    # train1 = scaler.fit_transform(train_data)
    train1 = train_data.reshape((train_data.shape[0],10,20))

    train_ = []
    for i in range(int(train1.shape[0]/(N+4))):
        for j in range(lags, N + 3):
            train_.append(train1[(N + 4) * i + (j - lags): (N + 4) * i + (j + 2)])
    train_ = np.array(train_)

    X_train = train_[:, :-2]
    X_train = moveaxis(X_train, 1, 3)
    y_train1 = train_[:, -2:-1].reshape((train_.shape[0], train_.shape[2] * train_.shape[3]))
    y_train2 = train_[:, -1:].reshape((train_.shape[0], train_.shape[2] * train_.shape[3]))

    print('X_train:', X_train.shape)
    return X_train, y_train1, y_train2

# get temporal features and future taxi demand (i+1, i+2 time interval)
def data_process_Corlstm(train_data):
    train_data = np.array(train_data)
    print(train_data.shape)

    # 归一化
    # train_data = scaler.fit_transform(train_data)

    train1 = []
    for i in range(int(train_data.shape[0]/(N+4))):
        x = train_data[i*(N+4):(i+1)*(N+4),:]
        x = x.transpose((1,0)).reshape(-1,1)
        train1.append(x)
    train1 = np.array(train1).reshape(-1,1)

    train_ = []
    for i in range(int(train1.shape[0] / (N+4))):
        for j in range(lags, N+3):
            train_.append(train1[(N+4) * i + (j - lags): (N+4) * i + (j + 2)])

    train_ = np.array(train_)
    train_ = train_.reshape((train_.shape[0], train_.shape[1]))

    X_train = train_[:, :-2]  # 用前lag个当做输入预测下一个时间点
    y_train1 = train_[:, -2:-1]
    y_train2 = train_[:, -1:]

    print('X_tain:', X_train.shape)
    print('y_train2', y_train2.shape)

    return X_train

# spatiotemporal features (random spatial + temporal + average)
def features_Cor(x_data, taxi_ave_demand):
    M = (N - 1) * 200  # the number of record in one day
    input_data = []

    # 3 correlation spatial features
    spatial_zone = pd.read_csv(file_1, header=0).iloc[:, 1:5]
    dic = {j : i for i, j in enumerate(spatial_zone.iloc[:, 0])}
    print(dic)
    spatial_cor = []
    for i in range(spatial_zone.shape[1]):
        if (i == 0):
            spatial_cor = np.array(list(map(dic.get, spatial_zone.iloc[:, i]))).reshape(-1, 1)
        else:
            x = np.array(list(map(dic.get, spatial_zone.iloc[:, i]))).reshape(-1, 1)
            spatial_cor = np.hstack((spatial_cor, x))
    # print(spatial_cor)

    for i in range(len(x_data)):
        x = []
        for j in range(lags):
            x.append(x_data[i][j])
        day = i//M
        zone = (i // (N -1))%200  #the k_th zone
        interval = i % (N-1)  #the m_th time interval

        for j in range(3):
            x.append(x_data[M * day + (N-1) * spatial_cor[zone][j+1] + interval][lags - 1])

        # average taxi demand feature
        x.append(taxi_ave_demand[interval][zone])
        input_data.append(x)
    input_data = np.array(input_data)
    print('input_data.shape', input_data.shape)
    return input_data

# get base models results
def get_base_model(X_data1):
    #  process data
    X_data = data_process_Corlstm(all_data)
    print('X_data', X_data.shape)
    X_data2 = features_Cor(X_data, taxi_ave_demand)

    X_data3 = X_data[:, :, np.newaxis]
    print('X_data3', X_data3.shape)

    # get base model result
    model1 =load_model('../model_save/cnn_{}_model.h5'.format(pred))
    y_predict1 = model1.predict(X_data1)
    print('predict1', y_predict1.shape)

    model2 = load_model('../model_save/corNet_{}_model.h5'.format(pred))
    y_predict2 = model2.predict(X_data2)
    print('predict2', y_predict2.shape)

    model3 = load_model('../model_save/lstm_{}_model.h5'.format(pred))
    y_predict3 = model3.predict(X_data3)
    print('predict3', y_predict3.shape)

    y_predict2_, y_predict3_ = [], []
    for i in range(int(y_predict2.shape[0]/(200*(N-1)))):
        x2 = y_predict2[i*200*(N-1):(i+1)*200*(N-1),:]
        x3 = y_predict3[i*200*(N-1):(i+1)*200*(N-1),:]

        # print('x2.shape', x2.shape)
        # print('x3.shape', x3.shape)

        x2 = x2.reshape((200, N-1)).T
        x3 = x3.reshape((200, N-1)).T

        if (i == 0):
            y_predict2_ = x2
            y_predict3_ = x3
        else:
            y_predict2_ = np.vstack((y_predict2_, x2))
            y_predict3_ = np.vstack((y_predict3_, x3))

    y_predict1 = scaler.inverse_transform(y_predict1)
    y_predict2_ = scaler.inverse_transform(y_predict2_)
    y_predict3_ = scaler.inverse_transform(y_predict3_)

    print('predict2', y_predict2_.shape)
    print('predict3', y_predict3_.shape)

    # print('predict1:\n%s'%(y_predict1))
    # print('predict1:\n%s' % (y_predict2_))
    # print('predict1:\n%s' % (y_predict3_))

    return y_predict1, y_predict2_, y_predict3_

# concatenating the features of base models
def feature_concatenate(X_feature1, X_feature2, X_feature3):
    X_feature1 = X_feature1.reshape((X_feature1.shape[0], 10, 20))
    X_feature2 = X_feature2.reshape((X_feature2.shape[0], 10, 20))
    X_feature3 = X_feature3.reshape((X_feature3.shape[0], 10, 20))

    X_features = []
    for i in range(X_feature1.shape[0]):
        x = []
        # print(X_feature1[i].shape)
        x.append(X_feature1[i])
        x.append(X_feature2[i])
        x.append(X_feature3[i])
        X_features.append(x)

    X_features = np.array(X_features, dtype=float)
    X_features = moveaxis(X_features, 1, 3)
    print('X_features:', X_features.shape)
    # print('X_features:\n%s'%(X_features))

    return X_features

# dateset split
def dataset_split(X_features, y_label1, y_label2):
    X_train = X_features[:(len(list_name) - 1) * (N - 1), ]
    X_test = X_features[(len(list_name) - 1) * (N - 1):, ]

    y_train1 = y_label1[:(len(list_name) - 1) * (N - 1), ]
    y_test1 = y_label1[(len(list_name) - 1) * (N - 1):, ]

    y_train2 = y_label2[:(len(list_name) - 1) * (N - 1), ]
    y_test2 = y_label2[(len(list_name) - 1) * (N - 1):, ]

    return X_train, y_train1, y_train2, X_test, y_test1, y_test2

# ——————————————— model building ————————————————————
# build network
def build_model(n1, n2, n3):
    model = Sequential([
        Conv2D(16, (3, 3), strides = 1, padding = 'same', activation='relu', input_shape=(n1, n2, n3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), strides = 1, padding = 'same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=256, activation='relu'),
        # Dense(units=n1*n2, activation='linear')
        Dense(units=n1 * n2, activation='sigmoid')
    ])

    model.summary()
    return model

# model training
def train_model(train_x, train_y):
    n1 = train_x.shape[1]
    n2 = train_x.shape[2]
    n3 = train_x.shape[3]
    model = build_model(n1, n2, n3)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='rme', optimizer=sgd)
    model.compile(loss='mse', optimizer='rmsprop')

    history = model.fit(train_x, train_y, batch_size = 32, epochs = 50, validation_split=0.1, verbose=1)

    loss = np.array(history.history['loss'])
    data1 = pd.DataFrame(loss)
    data1.to_csv('../../code result/train_loss/EnsemNet/loss_EnsembleNet-{}min.csv'.format(pred))

    plt.figure(1)
    plt.plot(history.history['loss'], 'b-')
    plt.legend(['loss'])
    plt.title('Model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    model.save('../model_save/ensemble_model.h5')

# model prediction
def predict_model(X_test, y_test):
    model = load_model('../model_save/ensemble_model.h5')
    predict = model.predict(X_test)

    print('predict', predict.shape)
    print('y_test', y_test.shape)


    #upscaled
    predict = scaler.inverse_transform(predict)
    y_test = scaler.inverse_transform(y_test)

    # MSE, MAE
    MSE = metrics.mean_squared_error(y_test, predict)
    MAE = metrics.mean_absolute_error(y_test, predict)

    print('MSE: %.3f' % (MSE))
    print('MAE: %.3f' % (MAE))

    print(predict)
    print(y_test)

    predict = predict.T
    return predict

if __name__ == '__main__':
    start = time.process_time()

    # number of time interval in one day
    N = int(60 / time_interval) * 17

    # obtain name of data file
    list_name = []
    eachFile(file_name, list_name)
    print(list_name)

    # get taxi demand and dataset
    taxi_ave_demand, all_data = get_taxi_demand(list_name)
    scaler = MinMaxScaler(feature_range=(0, 1))
    all_data = scaler.fit_transform(all_data)

    # get base model features
    X_data1, y_label1, y_label2 = data_process_CNN(all_data)
    X_feature1, X_feature2, X_feature3 = get_base_model(X_data1)
    X_features = feature_concatenate(X_feature1, X_feature2, X_feature3)
    print('X_features', X_features.shape)

    X_train, y_train1, y_train2, X_test, y_test1, y_test2 = dataset_split(X_features, y_label1, y_label2)

    print('X_test', X_test.shape)
    print('y_test1', y_test1.shape)
    print('y_test2', y_test2.shape)


    # model training
    train_model(X_train, y_train2)

    # model prediction
    # y_predict = predict_model(X_test, y_test2)

    # data = pd.DataFrame(y_predict)
    # data.to_csv('../../code result/10min/0-10min/prediction_10_EnsembNet.csv')
    # data.to_csv('../../code result/10min/10-20min/prediction_20_EnsemNet.csv')
    # data.to_csv('../../code result/15min/0-15min/prediction_15_EnsemNet.csv')
    # data.to_csv('../../code result/15min/15-30min/prediction_30_EnsemNet.csv')

    end = time.process_time()
    print('Running time: %s Seconds' % (end - start))
