# -*- coding: utf-8 -*-
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import os
import numpy.linalg as la
from utils import calculate_laplacian
from load_data import *
from tgcn import tgcnCell
from sklearn import metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pylab as plt
import time
import json

file_name = '../../data/zone_taxi_demand/10min_interval'
predict = 15

###### Settings ######
def parser():
    parser = argparse.ArgumentParser(description = 'variable about parameter')
    parser.add_argument('--lr',type= float, default =0.01)
    parser.add_argument('--epoch',type= int, default = 50)
    parser.add_argument('--gru_units', type= int, default = 64)
    parser.add_argument('--lag', type=int, default=4)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--validset_rate', type=float, default=0.1)
    parser.add_argument('--model_name',type= str,default='tgcn')
    return parser.parse_args()


def TGCN(_X, _weights, _biases):
    ###
    cell_1 = tgcnCell(args.gru_units, adj, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
    m = []
    for i in outputs:
        o = tf.reshape(i,shape=[-1,num_nodes,args.gru_units])
        o = tf.reshape(o,shape=[-1,args.gru_units])
        m.append(o)
    last_output = m[-1]
    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    output = tf.reshape(output,shape=[-1,num_nodes, args.pred_len])
    output = tf.transpose(output, perm=[0,2,1])
    output = tf.reshape(output, shape=[-1,num_nodes])
    return output, m, states


    
###### evaluation ######
def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a-b,'fro')/la.norm(a,'fro')
    r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    var = 1-(np.var(a-b))/np.var(a)
    return rmse, mae, 1-F_norm, r2, var


def train_model(trainX, trainY, testX, testY):
    # 训练集、验证集划分
    X_train, X_val, y_train, y_val = train_test_split(trainX, trainY, test_size = args.validset_rate, random_state = 42)
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)


    num_nodes = X_train.shape[2]
    batch_size = args.batch_size
    totalbatch = int(X_train.shape[0] / batch_size)


    ###### placeholders ######
    inputs = tf.placeholder(tf.float32, shape=[None, args.lag, num_nodes])
    labels = tf.placeholder(tf.float32, shape=[None, args.pred_len, num_nodes])

    # Graph weights
    weights = {'out': tf.Variable(tf.random_normal([args.gru_units, args.pred_len], mean=1.0), name='weight_o')}
    biases = {'out': tf.Variable(tf.random_normal([args.pred_len]), name='bias_o')}

    if args.model_name == 'tgcn':
        pred, ttts, ttto = TGCN(inputs, weights, biases)
    y_pred = pred

    ###### optimizer ######
    lambda_loss = 0.0015
    Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    label = tf.reshape(labels, [-1, num_nodes])
    ##loss
    loss = tf.reduce_mean(tf.nn.l2_loss(y_pred - label) + Lreg)

    ##rmse
    # error = tf.sqrt(tf.reduce_mean(tf.square(y_pred - label)))
    error = tf.reduce_mean(tf.square(y_pred - label))

    ##optimizer
    optimizer = tf.train.AdamOptimizer(args.lr).minimize(loss)

    ###### Initialize session ######
    # variables = tf.global_variables()
    saver = tf.train.Saver(tf.global_variables())

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    out = 'out/%s' % (args.model_name)
    path1 = '%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r' % (
        args.model_name, args.lr, args.batch_size, args.gru_units, args.lag, args.pred_len, args.epoch)
    path = os.path.join(out, path1)
    if not os.path.exists(path):
        os.makedirs(path)


    batch_loss, batch_rmse = [], []
    val_loss, val_rmse, val_mae, val_acc, val_r2, val_var, val_pred = [], [], [], [], [], [], []


    for epoch in range(args.epoch):
        time_start = time.time()

        for m in range(totalbatch):
            mini_batch = X_train[m * batch_size: (m + 1) * batch_size]
            mini_label = y_train[m * batch_size: (m + 1) * batch_size]
            _, loss1, rmse1, train_output = sess.run([optimizer, loss, error, y_pred],
                                                     feed_dict={inputs: mini_batch, labels: mini_label})

            print('Iter:{}/{}'.format(epoch, args.epoch),
                  'train_loss:{:.4}'.format(loss1),
                  'train_rmse:{:.4}'.format(rmse1),
                  'time:{:.4}'.format(time.time() - time_start))

        batch_loss.append(loss1)
        batch_rmse.append(rmse1)

        # val set performance at every epoch
        loss2, rmse2, val_output = sess.run([loss, error, y_pred],
                                             feed_dict={inputs: X_val, labels: y_val})

        # print('val_output:\n%s'%(val_output))
        print(val_output.shape)
        val_label = np.reshape(y_val, [-1, num_nodes])

        val_label = scaler.inverse_transform(val_label)
        val_output = scaler.inverse_transform(val_output)

        rmse0, mae, acc, r2_score, var_score = evaluation(val_label, val_output)


        val_loss.append(loss2)
        val_rmse.append(rmse0)
        val_mae.append(mae)
        val_acc.append(acc)
        val_r2.append(r2_score)
        val_var.append(var_score)

        print('Iter:{}/{}'.format(epoch, args.epoch),
              'val_loss:{:.4}'.format(loss2),
              'val_mae:{:.4}'.format(mae),
              'val_rmse:{:.4}'.format(rmse0),
              'val_acc:{:.4}'.format(acc),
              'time:{:.4}'.format(time.time()-time_start))

        if (epoch % 500 == 0):
            saver.save(sess, path + '/model_100/TGCN_pre_%r' % epoch, global_step=epoch)

    data1 = pd.DataFrame(np.array(batch_rmse))
    data1.to_csv('../../code result/train_loss/TGCN/loss_{}_TGCN.csv'.format(predict))

    print('loss:\t%s'%(batch_rmse))
    plt.figure(1)
    plt.plot(batch_rmse)
    plt.show()

    # Test completely at every epoch
    loss, test_output = sess.run([loss, y_pred], feed_dict={inputs: testX, labels: testY})
    print('test_output:\n%s' % (test_output))
    print(test_output.shape)

    test_label = np.reshape(testY, [-1, num_nodes])
    print('y_test', test_label.shape)

    y_test = scaler.inverse_transform(test_label)
    y_predict = scaler.inverse_transform(test_output)


    # MSE, MAE
    MSE = metrics.mean_squared_error(y_test, y_predict)
    MAE = metrics.mean_absolute_error(y_test, y_predict)

    print('MSE: %.3f' % (MSE))
    print('MAE: %.3f' % (MAE))


    print('y_predict:\n%s'%(y_predict))
    print('y_test:\n%s'%(y_test))
    return y_predict.T




def eval_model():
    pass



if __name__ == "__main__":
    args = parser()
    scaler = MinMaxScaler(feature_range=(0, 1))

    ###### load data ######

    train_data, test_data, adj = get_taxi_demand(file_name)
    num_nodes = train_data.shape[1]


    #### normalization
    adj = calculate_laplacian(adj)
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.fit_transform(test_data)

    X_train, y_train1, y_train2, X_test, y_test1, y_test2 = preprocess_data(train_data, test_data, args.lag)


    print('trainX', X_train.shape)
    print('trainY', y_train1.shape)
    print('testX', X_test.shape)
    print('y_test1', y_test1.shape)
    print('y_test2', y_test2.shape)

    start = time.clock()
    y_predict = train_model(X_train, y_train2, X_test, y_test2)
    data = pd.DataFrame(y_predict)

    print('train time:%s'%(time.clock()-start))
    # data.to_csv('../../code result/10min/0-10min/prediction_10_TGNN.csv')
    # data.to_csv('../../code result/10min/10-20min/prediction_20_TGCN.csv')
    # data.to_csv('../../code result/15min/0-15min/prediction_15_TGCN.csv')
    # data.to_csv('../../code result/15min/15-30min/prediction_30_TGCN.csv')

