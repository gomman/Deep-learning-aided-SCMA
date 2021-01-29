
import tensorflow as tf
import random
import numpy as np
import time
from pylab import *
# import matplotlib.pyplot as plt
import shutil
import os

from tensorflow.examples.tutorials.mnist import input_data

#tf.set_random_seed(777)  # reproducibility
tic = time.time()
#P = np.array([[0,1,1,0,0,1], [1,0,1,0,1,0], [1,0,0,1,0,1], [0,1,0,1,1,0]])
#res_ue_num = np.array([[1,2,3], [1,4,5],[2,4,6], [3,5,6]])
# parameters
learning_rate = 0.0001
#training_epochs = 1000000
batch_size = 200
modulation_level =4
# input place holders
X = tf.placeholder(tf.float32, [None, 24])
Y = tf.placeholder(tf.float32, [None, 24])
corruption = tf.placeholder(tf.float32,[None, 8])

# dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)
# X_1 = tf.transpose(tf.gather(tf.transpose(X),[4,5,6,7,8,9,10,11,16,17,18,19]))
# X_2 = tf.transpose(tf.gather(tf.transpose(X),[0,1,2,3,8,9,10,11,20,21,22,23]))
# X_3 = tf.transpose(tf.gather(tf.transpose(X),[4,5,6,7,12,13,14,15,20,21,22,23]))
# X_4 = tf.transpose(tf.gather(tf.transpose(X),[0,1,2,3,12,13,14,15,16,17,18,19]))
X_1 = tf.transpose(tf.gather(tf.transpose(X),[0,1,2,3]))
X_2 = tf.transpose(tf.gather(tf.transpose(X),[4,5,6,7]))
X_3 = tf.transpose(tf.gather(tf.transpose(X),[8,9,10,11]))
X_4 = tf.transpose(tf.gather(tf.transpose(X),[12,13,14,15]))
X_5 = tf.transpose(tf.gather(tf.transpose(X),[16,17,18,19]))
X_6 = tf.transpose(tf.gather(tf.transpose(X),[20,21,22,23]))

# resource 1 network
W21_1 = tf.get_variable("W21_1", shape=[4, 32], initializer=tf.contrib.layers.xavier_initializer())
b21_1 = tf.Variable(tf.random_normal([32]))
L21_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X_2, W21_1) + b21_1), keep_prob=keep_prob)
W21_2 = tf.get_variable("W21_2", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b21_2 = tf.Variable(tf.random_normal([32]))
L21_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(L21_1, W21_2) + b21_2), keep_prob=keep_prob)
W21_3 = tf.get_variable("W21_3", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b21_3 = tf.Variable(tf.random_normal([32]))
L21_3 = tf.nn.dropout(tf.nn.relu(tf.matmul(L21_2, W21_3) + b21_3), keep_prob=keep_prob)
W21_4 = tf.get_variable("W21_4", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b21_4 = tf.Variable(tf.random_normal([32]))
L21_4 = tf.nn.dropout(tf.nn.relu(tf.matmul(L21_3, W21_4) + b21_4), keep_prob=keep_prob)
W21_5 = tf.get_variable("W21_5", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b21_5 = tf.Variable(tf.random_normal([32]))
L21_5 = tf.nn.dropout(tf.nn.relu(tf.matmul(L21_4, W21_5) + b21_5), keep_prob=keep_prob)
W21_6 = tf.get_variable("W21_6", shape=[32, 2], initializer=tf.contrib.layers.xavier_initializer())
b21_6 = tf.Variable(tf.random_normal([2]))

W31_1 = tf.get_variable("W31_1", shape=[4, 32], initializer=tf.contrib.layers.xavier_initializer())
b31_1 = tf.Variable(tf.random_normal([32]))
L31_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X_3, W31_1) + b31_1), keep_prob=keep_prob)
W31_2 = tf.get_variable("W31_2", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b31_2 = tf.Variable(tf.random_normal([32]))
L31_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(L31_1, W31_2) + b31_2), keep_prob=keep_prob)
W31_3 = tf.get_variable("W31_3", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b31_3 = tf.Variable(tf.random_normal([32]))
L31_3 = tf.nn.dropout(tf.nn.relu(tf.matmul(L31_2, W31_3) + b31_3), keep_prob=keep_prob)
W31_4 = tf.get_variable("W31_4", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b31_4 = tf.Variable(tf.random_normal([32]))
L31_4 = tf.nn.dropout(tf.nn.relu(tf.matmul(L31_3, W31_4) + b31_4), keep_prob=keep_prob)
W31_5 = tf.get_variable("W31_5", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b31_5 = tf.Variable(tf.random_normal([32]))
L31_5 = tf.nn.dropout(tf.nn.relu(tf.matmul(L31_4, W31_5) + b31_5), keep_prob=keep_prob)
W31_6 = tf.get_variable("W31_6", shape=[32, 2], initializer=tf.contrib.layers.xavier_initializer())
b31_6 = tf.Variable(tf.random_normal([2]))

W51_1 = tf.get_variable("W51_1", shape=[4, 32], initializer=tf.contrib.layers.xavier_initializer())
b51_1 = tf.Variable(tf.random_normal([32]))
L51_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X_5, W51_1) + b51_1), keep_prob=keep_prob)
W51_2 = tf.get_variable("W51_2", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b51_2 = tf.Variable(tf.random_normal([32]))
L51_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(L51_1, W51_2) + b51_2), keep_prob=keep_prob)
W51_3 = tf.get_variable("W51_3", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b51_3 = tf.Variable(tf.random_normal([32]))
L51_3 = tf.nn.dropout(tf.nn.relu(tf.matmul(L51_2, W51_3) + b51_3), keep_prob=keep_prob)
W51_4 = tf.get_variable("W51_4", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b51_4 = tf.Variable(tf.random_normal([32]))
L51_4 = tf.nn.dropout(tf.nn.relu(tf.matmul(L51_3, W51_4) + b51_4), keep_prob=keep_prob)
W51_5 = tf.get_variable("W51_5", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b51_5 = tf.Variable(tf.random_normal([32]))
L51_5 = tf.nn.dropout(tf.nn.relu(tf.matmul(L51_4, W51_5) + b51_5), keep_prob=keep_prob)
W51_6 = tf.get_variable("W51_6", shape=[32, 2], initializer=tf.contrib.layers.xavier_initializer())
b51_6 = tf.Variable(tf.random_normal([2]))
# resource 1 network end
# resource 2 network
W12_1 = tf.get_variable("W12_1", shape=[4, 32], initializer=tf.contrib.layers.xavier_initializer())
b12_1 = tf.Variable(tf.random_normal([32]))
L12_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X_1, W12_1) + b12_1), keep_prob=keep_prob)
W12_2 = tf.get_variable("W12_2", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b12_2 = tf.Variable(tf.random_normal([32]))
L12_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(L12_1, W12_2) + b12_2), keep_prob=keep_prob)
W12_3 = tf.get_variable("W12_3", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b12_3 = tf.Variable(tf.random_normal([32]))
L12_3 = tf.nn.dropout(tf.nn.relu(tf.matmul(L12_2, W12_3) + b12_3), keep_prob=keep_prob)
W12_4 = tf.get_variable("W12_4", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b12_4 = tf.Variable(tf.random_normal([32]))
L12_4 = tf.nn.dropout(tf.nn.relu(tf.matmul(L12_3, W12_4) + b12_4), keep_prob=keep_prob)
W12_5 = tf.get_variable("W12_5", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b12_5 = tf.Variable(tf.random_normal([32]))
L12_5 = tf.nn.dropout(tf.nn.relu(tf.matmul(L12_4, W12_5) + b12_5), keep_prob=keep_prob)
W12_6 = tf.get_variable("W12_6", shape=[32, 2], initializer=tf.contrib.layers.xavier_initializer())
b12_6 = tf.Variable(tf.random_normal([2]))

W32_1 = tf.get_variable("W32_1", shape=[4, 32], initializer=tf.contrib.layers.xavier_initializer())
b32_1 = tf.Variable(tf.random_normal([32]))
L32_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X_3, W32_1) + b32_1), keep_prob=keep_prob)
W32_2 = tf.get_variable("W32_2", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b32_2 = tf.Variable(tf.random_normal([32]))
L32_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(L32_1, W32_2) + b32_2), keep_prob=keep_prob)
W32_3 = tf.get_variable("W32_3", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b32_3 = tf.Variable(tf.random_normal([32]))
L32_3 = tf.nn.dropout(tf.nn.relu(tf.matmul(L32_2, W32_3) + b32_3), keep_prob=keep_prob)
W32_4 = tf.get_variable("W32_4", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b32_4 = tf.Variable(tf.random_normal([32]))
L32_4 = tf.nn.dropout(tf.nn.relu(tf.matmul(L32_3, W32_4) + b32_4), keep_prob=keep_prob)
W32_5 = tf.get_variable("W32_5", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b32_5 = tf.Variable(tf.random_normal([32]))
L32_5 = tf.nn.dropout(tf.nn.relu(tf.matmul(L32_4, W32_5) + b32_5), keep_prob=keep_prob)
W32_6 = tf.get_variable("W32_6", shape=[32, 2], initializer=tf.contrib.layers.xavier_initializer())
b32_6 = tf.Variable(tf.random_normal([2]))

W62_1 = tf.get_variable("W62_1", shape=[4, 32], initializer=tf.contrib.layers.xavier_initializer())
b62_1 = tf.Variable(tf.random_normal([32]))
L62_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X_6, W62_1) + b62_1), keep_prob=keep_prob)
W62_2 = tf.get_variable("W62_2", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b62_2 = tf.Variable(tf.random_normal([32]))
L62_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(L62_1, W62_2) + b62_2), keep_prob=keep_prob)
W62_3 = tf.get_variable("W62_3", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b62_3 = tf.Variable(tf.random_normal([32]))
L62_3 = tf.nn.dropout(tf.nn.relu(tf.matmul(L62_2, W62_3) + b62_3), keep_prob=keep_prob)
W62_4 = tf.get_variable("W62_4", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b62_4 = tf.Variable(tf.random_normal([32]))
L62_4 = tf.nn.dropout(tf.nn.relu(tf.matmul(L62_3, W62_4) + b62_4), keep_prob=keep_prob)
W62_5 = tf.get_variable("W62_5", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b62_5 = tf.Variable(tf.random_normal([32]))
L62_5 = tf.nn.dropout(tf.nn.relu(tf.matmul(L62_4, W62_5) + b62_5), keep_prob=keep_prob)
W62_6 = tf.get_variable("W62_6", shape=[32, 2], initializer=tf.contrib.layers.xavier_initializer())
b62_6 = tf.Variable(tf.random_normal([2]))
# resource 2 network end
# resource 3 network
W23_1 = tf.get_variable("W23_1", shape=[4, 32], initializer=tf.contrib.layers.xavier_initializer())
b23_1 = tf.Variable(tf.random_normal([32]))
L23_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X_2, W23_1) + b23_1), keep_prob=keep_prob)
W23_2 = tf.get_variable("W23_2", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b23_2 = tf.Variable(tf.random_normal([32]))
L23_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(L23_1, W23_2) + b23_2), keep_prob=keep_prob)
W23_3 = tf.get_variable("W23_3", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b23_3 = tf.Variable(tf.random_normal([32]))
L23_3 = tf.nn.dropout(tf.nn.relu(tf.matmul(L23_2, W23_3) + b23_3), keep_prob=keep_prob)
W23_4 = tf.get_variable("W23_4", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b23_4 = tf.Variable(tf.random_normal([32]))
L23_4 = tf.nn.dropout(tf.nn.relu(tf.matmul(L23_3, W23_4) + b23_4), keep_prob=keep_prob)
W23_5 = tf.get_variable("W23_5", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b23_5 = tf.Variable(tf.random_normal([32]))
L23_5 = tf.nn.dropout(tf.nn.relu(tf.matmul(L23_4, W23_5) + b23_5), keep_prob=keep_prob)
W23_6 = tf.get_variable("W23_6", shape=[32, 2], initializer=tf.contrib.layers.xavier_initializer())
b23_6 = tf.Variable(tf.random_normal([2]))

W43_1 = tf.get_variable("W43_1", shape=[4, 32], initializer=tf.contrib.layers.xavier_initializer())
b43_1 = tf.Variable(tf.random_normal([32]))
L43_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X_4, W43_1) + b43_1), keep_prob=keep_prob)
W43_2 = tf.get_variable("W43_2", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b43_2 = tf.Variable(tf.random_normal([32]))
L43_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(L43_1, W43_2) + b43_2), keep_prob=keep_prob)
W43_3 = tf.get_variable("W43_3", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b43_3 = tf.Variable(tf.random_normal([32]))
L43_3 = tf.nn.dropout(tf.nn.relu(tf.matmul(L43_2, W43_3) + b43_3), keep_prob=keep_prob)
W43_4 = tf.get_variable("W43_4", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b43_4 = tf.Variable(tf.random_normal([32]))
L43_4 = tf.nn.dropout(tf.nn.relu(tf.matmul(L43_3, W43_4) + b43_4), keep_prob=keep_prob)
W43_5 = tf.get_variable("W43_5", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b43_5 = tf.Variable(tf.random_normal([32]))
L43_5 = tf.nn.dropout(tf.nn.relu(tf.matmul(L43_4, W43_5) + b43_5), keep_prob=keep_prob)
W43_6 = tf.get_variable("W43_6", shape=[32, 2], initializer=tf.contrib.layers.xavier_initializer())
b43_6 = tf.Variable(tf.random_normal([2]))

W63_1 = tf.get_variable("W63_1", shape=[4, 32], initializer=tf.contrib.layers.xavier_initializer())
b63_1 = tf.Variable(tf.random_normal([32]))
L63_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X_6, W63_1) + b63_1), keep_prob=keep_prob)
W63_2 = tf.get_variable("W63_2", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b63_2 = tf.Variable(tf.random_normal([32]))
L63_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(L63_1, W63_2) + b63_2), keep_prob=keep_prob)
W63_3 = tf.get_variable("W63_3", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b63_3 = tf.Variable(tf.random_normal([32]))
L63_3 = tf.nn.dropout(tf.nn.relu(tf.matmul(L63_2, W63_3) + b63_3), keep_prob=keep_prob)
W63_4 = tf.get_variable("W63_4", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b63_4 = tf.Variable(tf.random_normal([32]))
L63_4 = tf.nn.dropout(tf.nn.relu(tf.matmul(L63_3, W63_4) + b63_4), keep_prob=keep_prob)
W63_5 = tf.get_variable("W63_5", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b63_5 = tf.Variable(tf.random_normal([32]))
L63_5 = tf.nn.dropout(tf.nn.relu(tf.matmul(L63_4, W63_5) + b63_5), keep_prob=keep_prob)
W63_6 = tf.get_variable("W63_6", shape=[32, 2], initializer=tf.contrib.layers.xavier_initializer())
b63_6 = tf.Variable(tf.random_normal([2]))
# resource 3 network end
# resource 4 network
W14_1 = tf.get_variable("W14_1", shape=[4, 32], initializer=tf.contrib.layers.xavier_initializer())
b14_1 = tf.Variable(tf.random_normal([32]))
L14_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X_1, W14_1) + b14_1), keep_prob=keep_prob)
W14_2 = tf.get_variable("W14_2", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b14_2 = tf.Variable(tf.random_normal([32]))
L14_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(L14_1, W14_2) + b14_2), keep_prob=keep_prob)
W14_3 = tf.get_variable("W14_3", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b14_3 = tf.Variable(tf.random_normal([32]))
L14_3 = tf.nn.dropout(tf.nn.relu(tf.matmul(L14_2, W14_3) + b14_3), keep_prob=keep_prob)
W14_4 = tf.get_variable("W14_4", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b14_4 = tf.Variable(tf.random_normal([32]))
L14_4 = tf.nn.dropout(tf.nn.relu(tf.matmul(L14_3, W14_4) + b14_4), keep_prob=keep_prob)
W14_5 = tf.get_variable("W14_5", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b14_5 = tf.Variable(tf.random_normal([32]))
L14_5 = tf.nn.dropout(tf.nn.relu(tf.matmul(L14_4, W14_5) + b14_5), keep_prob=keep_prob)
W14_6 = tf.get_variable("W14_6", shape=[32, 2], initializer=tf.contrib.layers.xavier_initializer())
b14_6 = tf.Variable(tf.random_normal([2]))

W44_1 = tf.get_variable("W44_1", shape=[4, 32], initializer=tf.contrib.layers.xavier_initializer())
b44_1 = tf.Variable(tf.random_normal([32]))
L44_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X_4, W44_1) + b44_1), keep_prob=keep_prob)
W44_2 = tf.get_variable("W44_2", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b44_2 = tf.Variable(tf.random_normal([32]))
L44_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(L44_1, W44_2) + b44_2), keep_prob=keep_prob)
W44_3 = tf.get_variable("W44_3", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b44_3 = tf.Variable(tf.random_normal([32]))
L44_3 = tf.nn.dropout(tf.nn.relu(tf.matmul(L44_2, W44_3) + b44_3), keep_prob=keep_prob)
W44_4 = tf.get_variable("W44_4", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b44_4 = tf.Variable(tf.random_normal([32]))
L44_4 = tf.nn.dropout(tf.nn.relu(tf.matmul(L44_3, W44_4) + b44_4), keep_prob=keep_prob)
W44_5 = tf.get_variable("W44_5", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b44_5 = tf.Variable(tf.random_normal([32]))
L44_5 = tf.nn.dropout(tf.nn.relu(tf.matmul(L44_4, W44_5) + b44_5), keep_prob=keep_prob)
W44_6 = tf.get_variable("W44_6", shape=[32, 2], initializer=tf.contrib.layers.xavier_initializer())
b44_6 = tf.Variable(tf.random_normal([2]))

W54_1 = tf.get_variable("W54_1", shape=[4, 32], initializer=tf.contrib.layers.xavier_initializer())
b54_1 = tf.Variable(tf.random_normal([32]))
L54_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X_5, W54_1) + b54_1), keep_prob=keep_prob)
W54_2 = tf.get_variable("W54_2", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b54_2 = tf.Variable(tf.random_normal([32]))
L54_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(L54_1, W54_2) + b54_2), keep_prob=keep_prob)
W54_3 = tf.get_variable("W54_3", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b54_3 = tf.Variable(tf.random_normal([32]))
L54_3 = tf.nn.dropout(tf.nn.relu(tf.matmul(L54_2, W54_3) + b54_3), keep_prob=keep_prob)
W54_4 = tf.get_variable("W54_4", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b54_4 = tf.Variable(tf.random_normal([32]))
L54_4 = tf.nn.dropout(tf.nn.relu(tf.matmul(L54_3, W54_4) + b54_4), keep_prob=keep_prob)
W54_5 = tf.get_variable("W54_5", shape=[32, 32], initializer=tf.contrib.layers.xavier_initializer())
b54_5 = tf.Variable(tf.random_normal([32]))
L54_5 = tf.nn.dropout(tf.nn.relu(tf.matmul(L54_4, W54_5) + b54_5), keep_prob=keep_prob)
W54_6 = tf.get_variable("W54_6", shape=[32, 2], initializer=tf.contrib.layers.xavier_initializer())
b54_6 = tf.Variable(tf.random_normal([2]))
# resource 4 network end

S_1 = tf.add(tf.add(tf.matmul(L21_5, W21_6) + b21_6, tf.matmul(L31_5, W31_6) + b31_6), tf.matmul(L51_5, W51_6) + b51_6)
S_2 = tf.add(tf.add(tf.matmul(L12_5, W12_6) + b12_6, tf.matmul(L32_5, W32_6) + b32_6), tf.matmul(L62_5, W62_6) + b62_6)
S_3 = tf.add(tf.add(tf.matmul(L23_5, W23_6) + b23_6, tf.matmul(L43_5, W43_6) + b43_6), tf.matmul(L63_5, W63_6) + b63_6)
S_4 = tf.add(tf.add(tf.matmul(L14_5, W14_6) + b14_6, tf.matmul(L44_5, W44_6) + b44_6), tf.matmul(L54_5, W54_6) + b54_6)

stacked_symbol = tf.concat([S_1,S_2,S_3,S_4],1) # in version >R1.0 tf.concat([S_1,S_2,S_3,S_4],0)
encoded_symbol_normalizing = tf.sqrt(tf.reduce_mean(tf.square(stacked_symbol)))
#encoded_symbol_normalizing = tf.expand_dims(encoded_symbol_normalizing)
encoded_symbol_original = (1/np.sqrt(2))*tf.div(stacked_symbol, encoded_symbol_normalizing)
# encoded_symbol_normalizing_1 = tf.sqrt(tf.reduce_mean(tf.square(S_1),1))
# encoded_symbol_normalizing_2 = tf.sqrt(tf.reduce_mean(tf.square(S_2),1))
# encoded_symbol_normalizing_3 = tf.sqrt(tf.reduce_mean(tf.square(S_3),1))
# encoded_symbol_normalizing_4 = tf.sqrt(tf.reduce_mean(tf.square(S_4),1))
#
# encoded_symbol_normalizing_1 = tf.expand_dims(encoded_symbol_normalizing_1,1)
# encoded_symbol_normalizing_2 = tf.expand_dims(encoded_symbol_normalizing_2,1)
# encoded_symbol_normalizing_3 = tf.expand_dims(encoded_symbol_normalizing_3,1)
# encoded_symbol_normalizing_4 = tf.expand_dims(encoded_symbol_normalizing_4,1)
# encoded_symbol_original_1 = (1/np.sqrt(2))*tf.div(S_1, encoded_symbol_normalizing_1)
# encoded_symbol_original_2 = (1/np.sqrt(2))*tf.div(S_2, encoded_symbol_normalizing_2)
# encoded_symbol_original_3 = (1/np.sqrt(2))*tf.div(S_3, encoded_symbol_normalizing_3)
# encoded_symbol_original_4 = (1/np.sqrt(2))*tf.div(S_4, encoded_symbol_normalizing_4)
# encoded_symbol_original = tf.concat(1,[encoded_symbol_original_1,encoded_symbol_original_2,encoded_symbol_original_3,encoded_symbol_original_4])

encoded_symbol = encoded_symbol_original + corruption

W1 = tf.get_variable("W1", shape=[8, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(encoded_symbol, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[512, 24],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([24]))

hypothesis = tf.matmul(L3, W4) + b4

# define cost/loss & optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
cost = tf.reduce_mean(tf.square(hypothesis-Y))#+0.000001*(tf.nn.l2_loss(W1_1)+tf.nn.l2_loss(W2_1)+tf.nn.l2_loss(W3_1)+tf.nn.l2_loss(W4_1)+tf.nn.l2_loss(b1_1)+tf.nn.l2_loss(b2_1)+tf.nn.l2_loss(b3_1)+tf.nn.l2_loss(b4_1))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize

sess = tf.Session()
sess.run(tf.initialize_all_variables())
figure()
constellation_extr11 = np.zeros((7,2))
constellation_extr12 = np.zeros((7,2))
constellation_extr21 = np.zeros((7,2))
constellation_extr22 = np.zeros((7,2))
constellation_extr31 = np.zeros((7,2))
constellation_extr32 = np.zeros((7,2))
constellation_extr41 = np.zeros((7,2))
constellation_extr42 = np.zeros((7,2))
constellation_extr51 = np.zeros((7,2))
constellation_extr52 = np.zeros((7,2))
constellation_extr61 = np.zeros((7,2))
constellation_extr62 = np.zeros((7,2))
for SNR_range in np.arange(10,11,5):
    # train my model
    shutil.rmtree('./dictionary_newcode/')
    os.makedirs('./dictionary_newcode/')
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("dictionary_newcode")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print ("Could not find old network weights")
    lin_space = np.arange(0,31,5)
    print (lin_space)
    cha_mag = 2.0
    err_rate =[]
    ber_rate =[]
    for iterN in range(len(lin_space)):

        EbN0dB = lin_space[iterN]
        N0 = 1.0/3.0/np.power(10.0, EbN0dB/10.0)
        cost_plot =[]
        if lin_space[iterN] == SNR_range:
            training_epochs = 200001#100+SNR_range*30000
            for epoch in range(training_epochs):
                avg_cost = 0
                batch_ys = np.random.randint(4, size=(batch_size, 6))
                batch_y = np.zeros((batch_size, 24))
                for n in range(batch_size):
                    for m in range(6):
                        batch_y[n, m * 4 + batch_ys[n, m]] = 1
                noise_batch_r = (np.sqrt(N0 / 2.0)) * np.random.normal(0.0, size=(batch_size, 4))
                noise_batch_i = (np.sqrt(N0 / 2.0)) * np.random.normal(0.0, size=(batch_size, 4))
                rly = np.random.rayleigh(cha_mag / 2, (batch_size, 4))
                corruption_r = np.divide(noise_batch_r, rly)
                corruption_i = np.divide(noise_batch_i, rly)
                corruption_batch = np.hstack((corruption_r, corruption_i))
                #print (batch_xs.shape)
                #print (batch_ys.shape)
                feed_dict = {X: batch_y, Y: batch_y, keep_prob: 1.0, corruption: corruption_batch}
                # ss = sess.run(stacked_symbol,feed_dict=feed_dict)
                # print 'ss shape',ss.shape
                #print 'test1',test1
                # test2 = sess.run(encoded_symbol_normalizing,feed_dict=feed_dict)
                # print 'test2',test2
                # test33 = sess.run(tf.reduce_mean(tf.square(encoded_symbol_original),1), feed_dict=feed_dict)
                # print 'test3', test33
                c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
                avg_cost += c
                if epoch % 1000 ==0:
                    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
                    cost_plot.append(avg_cost)
                if epoch % 10000 == 0:
                    saver.save(sess, 'dictionary_newcode/' + 'network' + '-SCMA', global_step=epoch)

        print('Learning Finished!')

    for iterN in range(len(lin_space)):
        message = np.zeros((batch_size, 4), dtype=complex)
        EbN0dB = lin_space[iterN]
        N0 = 1.0 / 3.0 / np.power(10.0, EbN0dB / 10.0)
        test_batch_size = 100000
        test_ys = np.random.randint(4, size=(test_batch_size,6))
        test_y = np.zeros((test_batch_size,24))
        for n in range(test_batch_size):
            for m in range(6):
                test_y[n, m * 4 + test_ys[n, m]] = 1
                #test_y[n, 0] = 1
        noise_batch_test_r = (np.sqrt(N0 / 2.0)) * np.random.normal(0.0, size=(test_batch_size, 4))
        noise_batch_test_i = (np.sqrt(N0 / 2.0)) * np.random.normal(0.0, size=(test_batch_size, 4))
        rly = np.random.rayleigh(cha_mag / 2, (test_batch_size, 4))
        corruption_r = np.divide(noise_batch_test_r, rly)
        corruption_i = np.divide(noise_batch_test_i, rly)
        corruption_test_batch = np.hstack((corruption_r, corruption_i))
        #test_xs = np.hstack((np.real(message_test), np.imag(message_test))) + (np.random.normal(0, 0.01, (test_batch_size,8)) + 1j*np.random.normal(0, 0.01, (test_batch_size,8)))/np.random.rayleigh(1.0)
        correct_prediction1 = tf.equal(tf.argmax(hypothesis[:, 0:4], 1), tf.argmax(Y[:, 0:4], 1))
        correct_prediction2 = tf.equal(tf.argmax(hypothesis[:, 4:8], 1), tf.argmax(Y[:, 4:8], 1))
        correct_prediction3 = tf.equal(tf.argmax(hypothesis[:, 8:12], 1), tf.argmax(Y[:, 8:12], 1))
        correct_prediction4 = tf.equal(tf.argmax(hypothesis[:, 12:16], 1), tf.argmax(Y[:, 12:16], 1))
        correct_prediction5 = tf.equal(tf.argmax(hypothesis[:, 16:20], 1), tf.argmax(Y[:, 16:20], 1))
        correct_prediction6 = tf.equal(tf.argmax(hypothesis[:, 20:24], 1), tf.argmax(Y[:, 20:24], 1))

        correct_prediction = [correct_prediction1, correct_prediction2, correct_prediction3, correct_prediction4, correct_prediction5, correct_prediction6]

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        bit_error = []
        graycoding = tf.constant([[False, False], [False, True], [True, True], [True, False]])
        feed_dict = {X: test_y, Y: test_y, keep_prob: 1.0, corruption: corruption_test_batch}
        for i in range(6):
            bit_error.append(tf.reduce_mean(tf.cast(tf.logical_xor(tf.gather(graycoding, tf.argmax(hypothesis[:, i * modulation_level:(i + 1) * modulation_level], 1)),
                                                                   tf.gather(graycoding, tf.argmax(Y[:, i * modulation_level:(i + 1) * modulation_level], 1))), tf.float32)))
        BER = sess.run(tf.reduce_mean(bit_error), feed_dict=feed_dict)
        ber_rate.append(BER)
        #test1 = sess.run(encoded_symbol_original, feed_dict={X: test_y, Y: test_y, keep_prob: 1.0,corruption: corruption_test_batch})
        #test1 = sess.run(tf.reduce_mean(tf.square(encoded_symbol_original),1), feed_dict={X: test_y, Y: test_y, keep_prob: 1.0, corruption: corruption_test_batch})
        #print test1
        #plot(test1[:, 1], test1[:, 2], "r", marker=".", linestyle='None')
        ### extract layer constellation ###
        normalizing_factor = sess.run(encoded_symbol_normalizing, feed_dict={X: test_y, Y: test_y, keep_prob: 1.0, corruption: corruption_test_batch})
        # S_1 = tf.add(tf.add(tf.matmul(L21_5, W21_6) + b21_6, tf.matmul(L31_5, W31_6) + b31_6), tf.matmul(L51_5, W51_6) + b51_6)
        # S_2 = tf.add(tf.add(tf.matmul(L12_5, W12_6) + b12_6, tf.matmul(L32_5, W32_6) + b32_6), tf.matmul(L62_5, W62_6) + b62_6)
        # S_3 = tf.add(tf.add(tf.matmul(L23_5, W23_6) + b23_6, tf.matmul(L43_5, W43_6) + b43_6), tf.matmul(L63_5, W63_6) + b63_6)
        # S_4 = tf.add(tf.add(tf.matmul(L14_5, W14_6) + b14_6, tf.matmul(L44_5, W44_6) + b44_6), tf.matmul(L54_5, W54_6) + b54_6)
        normalizing_factor = tf.constant(normalizing_factor, dtype=tf.float32)
        corruption_extr = np.random.normal(0, 1, (1, 8))
        extr_run = np.zeros((12, 4), dtype=complex)
        for n in range(4):
            extract_y = np.zeros((1, 24))
            extract_y[0, 0+n] = 1
            extract_y[0, 4+n] = 1
            extract_y[0, 8+n] = 1
            extract_y[0, 12+n] = 1
            extract_y[0, 16+n] = 1
            extract_y[0, 20+n] = 1

            temp = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L12_5, W12_6) + b12_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            extr_run[0,n] = temp[0,0] + 1j*temp[0,1]
            temp = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L14_5, W14_6) + b14_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            extr_run[1, n] = temp[0, 0] + 1j * temp[0, 1]
            temp = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L21_5, W21_6) + b21_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            extr_run[2, n] = temp[0, 0] + 1j * temp[0, 1]
            temp = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L23_5, W23_6) + b23_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            extr_run[3, n] = temp[0, 0] + 1j * temp[0, 1]
            temp = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L31_5, W31_6) + b31_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            extr_run[4, n] = temp[0, 0] + 1j * temp[0, 1]
            temp = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L32_5, W32_6) + b32_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            extr_run[5, n] = temp[0, 0] + 1j * temp[0, 1]
            temp = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L43_5, W43_6) + b43_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            extr_run[6, n] = temp[0, 0] + 1j * temp[0, 1]
            temp = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L44_5, W44_6) + b44_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            extr_run[7, n] = temp[0, 0] + 1j * temp[0, 1]
            temp = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L51_5, W51_6) + b51_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            extr_run[8, n] = temp[0, 0] + 1j * temp[0, 1]
            temp = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L54_5, W54_6) + b54_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            extr_run[9, n] = temp[0, 0] + 1j * temp[0, 1]
            temp = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L62_5, W62_6) + b62_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            extr_run[10, n] = temp[0, 0] + 1j * temp[0, 1]
            temp = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L63_5, W63_6) + b63_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            extr_run[11, n] = temp[0, 0] + 1j * temp[0, 1]
            # constellation_extr11[iterN, :] = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L12_5, W12_6) + b12_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            # constellation_extr12[iterN, :] = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L14_5, W14_6) + b14_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            #
            # constellation_extr21[iterN, :] = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L21_5, W21_6) + b21_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            # constellation_extr22[iterN, :] = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L23_5, W23_6) + b23_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            #
            # constellation_extr31[iterN, :] = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L31_5, W31_6) + b31_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            # constellation_extr32[iterN, :] = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L32_5, W32_6) + b32_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            #
            # constellation_extr41[iterN, :] = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L43_5, W43_6) + b43_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            # constellation_extr42[iterN, :] = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L44_5, W44_6) + b44_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            #
            # constellation_extr51[iterN, :] = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L51_5, W51_6) + b51_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            # constellation_extr52[iterN, :] = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L54_5, W54_6) + b54_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            #
            # constellation_extr61[iterN, :] = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L62_5, W62_6) + b62_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})
            # constellation_extr62[iterN, :] = sess.run((1 / np.sqrt(2)) * tf.div(tf.matmul(L63_5, W63_6) + b63_6, normalizing_factor), feed_dict={X: extract_y, Y: extract_y, keep_prob: 1.0, corruption: corruption_extr})

        np.savetxt("./dictionary_newcode/learned_constellations", extr_run)
        ### extract layer constellation ###
        SER = 1 - sess.run(accuracy, feed_dict={X: test_y, Y: test_y, keep_prob: 1.0,corruption: corruption_test_batch})
        err_rate.append(SER)
    np.savetxt("./dictionary_newcode/BER_trained_at_{0}dB_rly".format(SNR_range), ber_rate)

print (lin_space)
print (ber_rate)
show()
toc = time.time()
print("elapsed time", toc-tic)


# Get one and predict
# r = random.randint(0, mnist.test.num_examples - 1)
# print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
# print("Prediction: ", sess.run(
#     tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))

# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()
