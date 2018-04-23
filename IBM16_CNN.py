"""
This is an implementation of CNN architecture presented in "characterizing driving styles with deep learning". 
Author: Sobhan Moosavi
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
import math
from scipy import stats
import time

import cPickle
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


class CNN_MODEL:

    def __init__(self, data, target, dropout):
        self.data = data
        self.target = target
        self._dropout = dropout
        self.cost
        self.prediction
        self.optimize
        self.accuracy

    @lazy_property
    def prediction(self):    
        # Input Layer
        # Reshape X to 4-D tensor: [batch_size, width, height, channels]
        # Trajectory Segments are 35x128, and we just have one channel. 
        input_layer = tf.reshape(self.data, [-1, 35, 128, 1])

        # Convolutional Layer #1
        # Computes 32 features using a 35x5 filter with Sigmoid activation. [convolution is over time]
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 35, 128, 1]
        # Output Tensor Shape: [batch_size, 1 124, 32]
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[35, 5], strides=1, activation=tf.nn.sigmoid)

        # Pooling Layer #1
        # First max pooling layer with a 1x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 1, 124, 32]
        # Output Tensor Shape: [batch_size, 1, 62, 32]
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 2], strides=2)

        # Convolutional Layer #2
        # Computes 64 features using a 1x3 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 1, 62, 32]
        # Output Tensor Shape: [batch_size, 1, 60, 64]
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[1, 3], strides=1, activation=tf.nn.sigmoid)

        # Pooling Layer #2
        # Second max pooling layer with a 1x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 1, 60, 64]
        # Output Tensor Shape: [batch_size, 1, 30, 64]
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 2], strides=2)

        # Convolutional Layer #3
        # Computes 64 features using a 1x3 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 1, 30, 64]
        # Output Tensor Shape: [batch_size, 1, 28, 64]
        conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[1, 3], strides=1, activation=tf.nn.sigmoid)

        # Pooling Layer #3
        # Third max pooling layer with a 1x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 1, 28, 64]
        # Output Tensor Shape: [batch_size, 1, 14, 64]
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[1, 2], strides=2)

        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, 1, 14, 64]
        # Output Tensor Shape: [batch_size, 1 * 14 * 64]
        pool3_flat = tf.reshape(pool3, [-1, 1 * 14 * 64])

        # Dense Layer #1
        # Densely connected layer with 128 neurons
        # Input Tensor Shape: [batch_size, 7 * 7 * 64]
        # Output Tensor Shape: [batch_size, 128]
        dense1 = tf.layers.dense(inputs=pool3_flat, units=128, activation=tf.nn.sigmoid)

        # Dropout Layer #1
        # Add dropout operation; (1-rate) probability that element will be kept
        dropout1 = tf.layers.dropout(inputs=dense1, rate=self._dropout)

        # Dense Layer #2
        # Densely connected layer with 128 neurons
        # Input Tensor Shape: [batch_size, 7 * 7 * 64]
        # Output Tensor Shape: [batch_size, 128]
        dense2 = tf.layers.dense(inputs=dropout1, units=128, activation=tf.nn.sigmoid)

        # Dropout Layer #2
        # Add dropout operation; (1-rate) probability that element will be kept
        dropout2 = tf.layers.dropout(inputs=dense2, rate=self._dropout)

        # Logits layer
        # Input Tensor Shape: [batch_size, 128]
        # Output Tensor Shape: [batch_size, numOfDrivers]
        logits = tf.layers.dense(inputs=dropout2, units=int(self.target.get_shape()[1]), activation=None) 
                
        predicted_classes = tf.argmax(input=logits, axis=1)
        softmax_prob = tf.nn.softmax(logits, name="softmax_tensor")
        
        return logits, predicted_classes, softmax_prob
    
    @lazy_property
    def cost(self):
        logits, predicted_classes, softmax_prob = self.prediction               
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.target * tf.log(softmax_prob), reduction_indices=[1]))
        return cross_entropy

    @lazy_property
    def optimize(self):       
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.05, momentum=0.9, use_nesterov=True)
        return optimizer.minimize(self.cost)
    
    @lazy_property
    def accuracy(self):
        logits, predicted_classes, softmax_prob = self.prediction
        correct_pred = tf.equal(tf.argmax(self.target, 1), tf.argmax(softmax_prob, 1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def load_data(file):
    trip_segments = np.load(file)
    print("Number of samples: {}".format(trip_segments.shape[0]))
    return trip_segments
  
 
def returnTrainAndTestData():
    
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
        m = np.transpose(m) #need this step and the next for CNN
        m = np.reshape(m, 35*128)
        if r < .8:
          train_data.append(m)
          train_labels.append(dr)
        elif r < .87:
            dev_data.append(m)
            dev_labels.append(dr)
        else:
          test_data.append(m)
          test_labels.append(dr)        

    train_data   = np.asarray(train_data, dtype="float32")
    train_labels = np.asarray(train_labels, dtype="int32")
    dev_data     = np.asarray(dev_data, dtype="float32")
    dev_labels   = np.asarray(dev_labels, dtype="int32")
    test_data    = np.asarray(test_data, dtype="float32")
    test_labels  = np.asarray(test_labels, dtype="int32")

    train_data, train_labels = shuffle_in_union(train_data, train_labels)   #Does shuffling do any help ==> it does a great help!!
  
    return train_data, train_labels, dev_data, dev_labels, test_data, test_labels, len(driverIds)

    
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

    args = [50, 200]
    st = time.time()
    train, train_labels, dev, dev_labels, test, test_labels, num_classes = returnTrainAndTestData()
    print('All data is loaded in {:.1f} seconds'.format(time.time()-st))
    
    display_step = 100
    training_steps = 250000
    batch_size = 256
    
    train_dropout = 0.5
    test_dropout = 0.0
    
    timesteps = 128 # Number of rows in Matrix of a Segment
    
    train_labels = convertLabelsToOneHotVector(train_labels, num_classes)   
    dev_labels   = convertLabelsToOneHotVector(dev_labels, num_classes)
    test_labels  = convertLabelsToOneHotVector(test_labels, num_classes)
    
    data = tf.placeholder(tf.float32, [None, 35*128], name='data')    
    target = tf.placeholder(tf.float32, [None, num_classes], name='target')
    dropout = tf.placeholder(tf.float32)
    model = CNN_MODEL(data, target, dropout)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    train_start = time.time()
    start = time.time()
    next_batch_start = 0
    
    steps_to_epoch = len(train)/batch_size
    
    maxDevAccuracy = 0.0 #This will be used as a constraint to save the best model
    bestEpoch = 0
    
    saver = tf.train.Saver() #This is the saver of the model    
    
    for step in range(training_steps):
        idx_end = min(len(train),next_batch_start+batch_size)        
        sess.run(model.optimize, {data: train[next_batch_start:idx_end,:], target: train_labels[next_batch_start:idx_end,:], dropout: train_dropout})                
        
        epoch = int(step/steps_to_epoch)
        if epoch > bestEpoch or epoch == 0: 
            acc = sess.run(model.accuracy, {data: dev, target: dev_labels, dropout: test_dropout})
            if epoch > 5 and acc > maxDevAccuracy:
                maxDevAccuracy = acc
                bestEpoch = epoch
                save_path = saver.save(sess, 'models/bestCNN_{}_{}_B{}/'.format(args[0], args[1], batch_size))
                print('Model saved in path: {}, Dev Accuracy: {:.2f}%, Epoch: {:d}'.format(save_path, 100*acc, epoch))
        
        if step % display_step == 0:            
            loss = sess.run(model.cost, {data: train[next_batch_start:idx_end,:], target: train_labels[next_batch_start:idx_end,:], dropout: test_dropout})
            train_acc = sess.run(model.accuracy, {data: train[next_batch_start:idx_end,:], target: train_labels[next_batch_start:idx_end,:], dropout: test_dropout})
            dev_acc = sess.run(model.accuracy, {data: dev, target: dev_labels, dropout: test_dropout})
            dev_loss  = sess.run(model.cost, {data: dev, target: dev_labels, dropout: test_dropout})              
            print('Step {:2d}, Epoch {:2d}, Train Loss {:.3f}, Dev-Loss {:.3f}, Mini-Batch Train_Accuracy {:.1f}%, Dev-Accuracy {:.1f}%, ({:.1f} sec)'.format(step + 1, epoch, loss, dev_loss, 100*train_acc, 100*dev_acc, (time.time()-start)))            
            start = time.time()
            
        next_batch_start += next_batch_start+batch_size        
        if next_batch_start >= len(train):
            train, train_labels = shuffle_in_union(train, train_labels)
            next_batch_start = 0
    
    
    print("Optimization Finished!")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'models/bestCNN_{}_{}_B{}/'.format(args[0], args[1], batch_size))
    accuracy = sess.run(model.accuracy, {data: test, target: test_labels, dropout: test_dropout})
    print('Test-Accuracy: {:.2f}%, Train-Time: {:.1f}sec'.format(accuracy*100, (time.time()-train_start)))
    print('Best Dev-Accuracy: {:.2f}%, Best Epoch: {}'.format(maxDevAccuracy*100, bestEpoch))