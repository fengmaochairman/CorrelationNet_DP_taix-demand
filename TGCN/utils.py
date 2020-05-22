# -*- coding: utf-8 -*-

import tensorflow as tf
import scipy.sparse as sp
import numpy as np


def normalized_adj(adj):
    adj = sp.coo_matrix(adj)  #转换成 coo型稀疏矩阵
    rowsum = np.array(adj.sum(1))  # 对每一行求和
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # 构建对角元素为r_inv的对角矩阵, 即D^-0.5
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5*A*D^-0.5
    normalized_adj = normalized_adj.astype(np.float32)
    return normalized_adj
    
def sparse_to_tuple(mx):
    mx = mx.tocoo()  #按列向量形式
    coords = np.vstack((mx.row, mx.col)).transpose()  # np.vstack:垂直堆叠
    L = tf.SparseTensor(coords, mx.data, mx.shape) #转为稀疏张量
    return tf.sparse_reorder(L) 
    
def calculate_laplacian(adj, lambda_max=1):  
    adj = normalized_adj(adj + sp.eye(adj.shape[0]))
    adj = sp.csr_matrix(adj)
    adj = adj.astype(np.float32)
    return sparse_to_tuple(adj)
    
def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                            maxval=init_range, dtype=tf.float32)

    return tf.Variable(initial,name=name)  