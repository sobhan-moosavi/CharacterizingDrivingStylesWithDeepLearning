"""
This is an implementtion of RNN architecture, presented in "characterizing driving styles with deep learning".
Author: Sobhan Moosavi
"""

from __future__ import print_function
from __future__ import division
import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np
import random
import math
from scipy import stats
import time
import cPickle
import time

from sklearn.preprocessing import OneHotEncoder
import functools

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class SequenceClassification:

    def __init__(self, data, target, dropout, num_layers, num_hidden=100, timesteps=128):
        self.data = data
        self.target = target
        self.dropout = dropout
        self.num_layers = num_layers
        self._num_hidden = num_hidden        
        self._timesteps = timesteps
        self.prediction
        self.error
        self.optimize
        self.accuracy

    @lazy_property
    def prediction(self):
        # Recurrent network.
        stacked_rnn = []
        for i in range(self.num_layers):
            cell = rnn.BasicLSTMCell(num_units=self._num_hidden, state_is_tuple=True, forget_bias=1.0)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0-self.dropout[i])
            stacked_rnn.append(cell)            
        network = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
        
        #output, _ = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)
        x = tf.unstack(self.data, self._timesteps, 1)
        output, _ = rnn.static_rnn(network, x, dtype=tf.float32)
        
        # Softmax layer parameters
        weight, bias = self._weight_and_bias(self._num_hidden, int(self.target.get_shape()[1]))
        
        #Embedding
        embedding = tf.matmul(output[-1], tf.Variable(np.identity(self._num_hidden, dtype="float32")))
        
        # Linear activation, using rnn inner loop last output    
        logits = tf.matmul(output[-1], weight) + bias
        soft_reg = tf.nn.softmax(logits)
        #prediction = tf.nn.softmax(logits)        
        return soft_reg  
    
    @lazy_property
    def cost(self):
        #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.target))
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.target * tf.log(self.prediction), reduction_indices=[1]))  
        return cross_entropy

    @lazy_property
    def optimize(self):
        #optimizer = tf.train.RMSPropOptimizer(learning_rate=0.003)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00005, momentum=0.9, epsilon=1e-6)        
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))
    
    @lazy_property
    def accuracy(self):
        correct_pred = tf.equal(tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


#Following functions are related to feature encoding [creating the statistical feature matrix for each trajectory segment]

class point:
    lat = 0
    lng = 0
    time = 0
    def __init__(self, time, lat, lng):
        self.lat = lat
        self.lng = lng
        self.time = time

        
class basicFeature:
    speedNorm = 0
    diffSpeedNorm = 0
    accelNorm = 0
    diffAccelNorm = 0
    angularSpeed = 0
    def __init__(self, speedNorm, diffSpeedNorm, accelNorm, diffAccelNorm, angularSpeed):
        self.speedNorm = speedNorm
        self.diffSpeedNorm= diffSpeedNorm
        self.accelNorm= accelNorm
        self.diffAccelNorm = diffAccelNorm
        self.angularSpeed = angularSpeed


def load_data(file):
    trip_segments = np.load(file)#/40.0
    print("Number of samples: {}".format(trip_segments.shape[0]))
    return trip_segments
    """np.random.shuffle(trip_segments)
    split_idx = int((1-args.val_frac) * trip_segments.shape[0])
    return trip_segments[:split_idx], trip_segments[split_idx:]"""
        
        
def returnTrainDevTestData():
    
    matrices = load_data('data/smallSample_{}_{}.npy'.format(args[0], args[1]))
    keys = cPickle.load(open('data/smallSample_{}_{}_keys.pkl'.format(args[0], args[1]), 'rb'))        
    
    #Build Train, Dev, Test sets
    train_data = []
    train_labels = []
    dev_data = []
    dev_labels = []
    test_data = []
    test_labels = []
    
    curTraj = ''
    r = 0
    
    driverIds = {}
    
    for idx in range(len(keys)):
        d,t = keys[idx]
        if d in driverIds:
            dr = driverIds[d]
        else: 
            dr = len(driverIds)
            driverIds[d] = dr
        m = matrices[idx][1:129,]
        #print (d, t, idx, m.shape)    
        if t != curTraj:
            curTraj = t
            r = random.random()                    
        if m.shape[0] < 128:
          continue 
        if r < .8:
          train_data.append(m)
          train_labels.append(dr)
        elif r < .9:
          dev_data.append(m)
          dev_labels.append(dr)
        else:
          test_data.append(m)
          test_labels.append(dr)        

    train_data   = np.asarray(train_data, dtype="float32")
    train_labels = np.asarray(train_labels, dtype="int32")
    dev_data   = np.asarray(dev_data, dtype="float32")
    dev_labels = np.asarray(dev_labels, dtype="int32")
    test_data    = np.asarray(test_data, dtype="float32")
    test_labels  = np.asarray(test_labels, dtype="int32")

    train_data, train_labels = shuffle_in_union(train_data, train_labels)   #Does shuffling do any help ==> it does a great help!!
  
    return train_data, train_labels, dev_data, dev_labels, test_data, test_labels, len(driverIds)+1

    
def shuffle_in_union(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

    
def convertLabelsToOneHotVector(labels, ln):
    tmp_lb = np.reshape(labels, [-1,1])
    next_batch_start = 0
    _x = np.arange(ln)
    _x = np.reshape(_x, [-1, 1])
    enc = OneHotEncoder()
    enc.fit(_x)
    labels =  enc.transform(tmp_lb).toarray()
    return labels
  

if __name__ == '__main__':
    #Arguments to specify the data file for train and test. 
    args = [5, 5] # the first input shows number of drivers, and the second one the number of trajectories per driver
    st = time.time()
    train, train_labels, dev, dev_labels, test, test_labels, num_classes = returnTrainDevTestData()
    print('Train, Dev, Test datasets are loaded in {:.1f} seconds!'.format(time.time()-st))
    
    display_step = 100
    training_steps = 75000
    batch_size = 128
    
    timesteps = 128 # Number of rows in Matrix of a Segment
    num_layers = 2 # Number of network layers
    dropouts_train = [0.4, 0.6] #dropout values for different network layers [for train]
    dropouts_dev  = [0.0, 0.0] #dropout values for different network layers [for test and dev]

    train_labels = convertLabelsToOneHotVector(train_labels, num_classes)    
    dev_labels = convertLabelsToOneHotVector(dev_labels, num_classes)
    test_labels = convertLabelsToOneHotVector(test_labels, num_classes)
    
    #print(train.shape, dev.shape, test.shape)
    
    data = tf.placeholder(tf.float32, [None, 128, 35])    
    target = tf.placeholder(tf.float32, [None, num_classes])
    dropout = tf.placeholder(tf.float32, [len(dropouts_train)])
    model = SequenceClassification(data, target, dropout, num_layers)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    #print(train.shape)    
    
    train_start = time.time()
    start = time.time()
    next_batch_start = 0
    
    maxTestAccuracy = 0.0 #This will be used as a constraint to save the best model
    bestEpoch = 0    
    saver = tf.train.Saver() #This is the saver of the model    
    
    steps_to_epoch = len(train)/batch_size
    
    for step in range(training_steps):
        idx_end = min(len(train),next_batch_start+batch_size)        
        sess.run(model.optimize, {data: train[next_batch_start:idx_end,:], target: train_labels[next_batch_start:idx_end,:], dropout: dropouts_train})
        
        epoch = int(step/steps_to_epoch)
        if epoch > bestEpoch or epoch == 0:
            acc = sess.run(model.accuracy, {data: dev[0:min(8*batch_size, len(dev))], target: dev_labels[0:min(8*batch_size, len(dev))], dropout: dropouts_dev})
            if epoch > 5 and acc > maxTestAccuracy:
                maxTestAccuracy = acc
                bestEpoch = epoch
                save_path = saver.save(sess, 'models/bestRNN_{}_{}_DP4_6_B256/'.format(args[0], args[1]))
                print('Model saved in path: {}, Accuracy: {:.2f}%, Epoch: {:d}'.format(save_path, 100*acc, epoch))
        
        if step % display_step == 0:
            loss_train = sess.run(model.cost, {data: train[next_batch_start:idx_end,:], target: train_labels[next_batch_start:idx_end,:], dropout: dropouts_dev})
            loss_dev = sess.run(model.cost, {data: dev, target: dev_labels, dropout: dropouts_dev})
            acc_train  = sess.run(model.accuracy, {data: train[next_batch_start:idx_end,:], target: train_labels[next_batch_start:idx_end,:], dropout: dropouts_dev})
            acc_dev  = sess.run(model.accuracy, {data: dev, target: dev_labels, dropout: dropouts_dev})
            print('Step {:2d}, Epoch {:2d}, Minibatch Train Loss {:.3f}, Dev Loss {:.3f}, Train-Accuracy {:.1f}%, Dev-Accuracy {:.1f}% ({:.1f} sec)'.format(step + 1, epoch, loss_train, loss_dev, 100 * acc_train, 100*acc_dev, (time.time()-start)))
            start = time.time()
        next_batch_start += batch_size
        if next_batch_start >= len(train):
            train, train_labels = shuffle_in_union(train, train_labels)
            dev, dev_labels = shuffle_in_union(dev, dev_labels)
            next_batch_start = 0
    
    print("Optimization Finished!")    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'models/bestRNN_{}_{}_DP4_6_B256/'.format(args[0], args[1]))
    accuracy = sess.run(model.accuracy, {data: test, target: test_labels, dropout: dropouts_dev})
    print('Final Test-Accuracy: {:.2f}%, Train-Time: {:.1f}sec'.format(accuracy*100, (time.time()-train_start)))
    print('Partial Best Test-Accuracy: {:.2f}%, Best Epoch: {}'.format(maxTestAccuracy*100, bestEpoch))
