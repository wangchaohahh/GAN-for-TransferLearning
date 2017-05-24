# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 18:27:15 2017

@author: wangchao
"""

#from scipy.misc import imread
#import pylab as pl
#
#img = imread('.//dataset//amazon//images//back_pack//frame_0001.jpg')
#pl.imshow(img)

#import tools
#
#tools.test_load()

#from tensorflow.contrib import slim
#import tensorflow as tf
#
#with tf.variable_scope('conv1'):
#    weights2 = slim.model_variable('weights4',
#                              shape=[10, 10, 3 , 3],
#                              initializer=tf.truncated_normal_initializer(stddev=0.1),
#                              regularizer=slim.l2_regularizer(0.05),
#                              )
#
#model_variables = slim.get_model_variables()
#print (model_variables)

trainable = ['a1', 'b0', 'c2']
g_vars = [v for v in trainable if 'a1' in v]
d_vars = [v for v in trainable if 'b0' in v]
e_vars = [v for v in trainable if v not in g_vars and v not in d_vars]
print (e_vars)

























