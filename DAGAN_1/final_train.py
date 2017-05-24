# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 19:51:21 2017

@author: wangchao
"""

import tensorflow as tf
import vaegan
from input_data import read_and_decode
import tools
import time


## parameters
IMG_W = 224
IMG_H = 224
N_CLASSES = 10
BATCH_SIZE = 100
learning_rate = 0.001
MAX_STEP = 200
IS_PRETRAIN = True
z_dim = 256
gamma = 0.01  ## reconst_v_gan trade off parameter
alpha = 0.01  ## reconst vs classifier trade off parameter
lamda = 0.01  ## target MMD vs GAN trade off parameter


### placeholder
#Xs = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
#Xt = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
#ys_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_CLASSES])
#yt_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_CLASSES])

## load source and target domain data
Xs_input, Ys_input = read_and_decode('/input/dslr.tfrecords',BATCH_SIZE)
Xt_input, Yt_input = read_and_decode('/input/webcam.tfrecords',BATCH_SIZE)
# Xt_test, Yt_test = read_and_decode('/input/webcam.tfrecords',BATCH_SIZE, is_batch= False)
####################### loss ###########################################

loss, yt_predict = vaegan.loss(Xs_input, Xt_input, Ys_input)

## source domain  loss
obj_D = loss['D_fake'] + loss['D_real']
obj_G = loss['G_fake'] + gamma*loss['Dis']
obj_E = loss['C_predict'] + (loss['KL(z)'] + loss['Dis'])*alpha

trainables = tf.trainable_variables()
g_vars = [v for v in trainables if 'Generator' in v.name]
d_vars = [v for v in trainables if 'Discriminator' in v.name]
e_vars = [v for v in trainables if v not in g_vars and v not in d_vars]

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
opt_e = optimizer.minimize(obj_E, var_list=e_vars)
opt_d = optimizer.minimize(obj_D, var_list=d_vars)
opt_g = optimizer.minimize(obj_G, var_list=g_vars)

## target domain  loss
opt_Target = loss['D_fake_t'] + lamda*loss['MMD']
opt_tar = optimizer.minimize(opt_Target, var_list=e_vars)

######################accuracy#############################
accuracy = tools.accuracy(yt_predict, Yt_input)
######################train################################

sess = tf.Session()
init = tf.global_variables_initializer()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

sess.run(init)
#tools.load_with_skip('/input/vgg16.npy', sess, ['fc6','fc7','fc8'])

try:
    for iter in range(MAX_STEP):
        # start_time = time.time()
        _, loss_D = sess.run([opt_d, obj_D])
    
        # updata G twice
        _, loss_G = sess.run([opt_g, obj_G])

        # _, loss_G = sess.run([opt_g, obj_G])
    
        _, loss_E = sess.run([opt_e, obj_E])
    
        ## update target domain
        _, loss_Tar = sess.run([opt_tar, opt_Target])
    	
        # time_iter = time.time()-start_time
    
        ## print message
        if iter % 1 == 0:
            # accuary
            acc = sess.run(accuracy)
            msg = 'Epoch{:3d}'.format(iter+1) + '>>>>'+ 'Acc{:6.3f}'.format(acc)
            print (msg)
except tf.errors.OutOfRangeError:
    print('done!')
finally:
    coord.request_stop()
    coord.join(threads)


















