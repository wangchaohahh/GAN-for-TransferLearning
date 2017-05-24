# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 18:22:59 2017

@author: wangchao
"""

from tensorflow.contrib import slim
import tensorflow as tf
import tools

def Encoder_VGG(x, z_dim, n_class, is_pretrain=True, reuse = False):
    if (reuse):
        tf.get_variable_scope().reuse_variables() 
    x = tools.conv('conv1_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv1_2', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.pool('pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
    
    x = tools.conv('conv2_1', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv2_2', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.pool('pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
    
    x = tools.conv('conv3_1', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv3_2', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv3_3', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
    
    x = tools.conv('conv4_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv4_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv4_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

    x = tools.conv('conv5_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv5_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv5_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)            

    x = tools.FC_layer('fc6', x, out_nodes=4096)
    x = tools.batch_norm(x)
    z_mu = tools.FC_layer('fc8', x, out_nodes=z_dim)
    z_lv = tools.FC_layer('fc9', x, out_nodes=z_dim)
    y_predict = tools.FC_layer('fc10', z_mu, out_nodes=n_class)
        

    return z_mu, z_lv, y_predict

def Encoder(x, z_dim, n_class):
    with slim.arg_scope(
            [slim.batch_norm],
            scale=True,
            updates_collections=None,
            decay=0.9, epsilon=1e-5,
            is_training=True,
            reuse=None):
        with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(1e-6),
                normalizer_fn=slim.batch_norm,
                activation_fn=tf.nn.relu):
            x = slim.conv2d(x,64,[5,5],[2,2])
            x = slim.conv2d(x, 128, [5, 5], [2, 2])
            x = slim.conv2d(x, 256, [5, 5], [2, 2])

    x = slim.flatten(x)
    with slim.arg_scope(
        [slim.fully_connected],
        num_outputs=z_dim,
        weights_regularizer=slim.l2_regularizer(1e-6),
        normalizer_fn=None,
        activation_fn=None):
        z_mu = slim.fully_connected(x)
        z_lv = slim.fully_connected(x)
        y_predict = slim.fully_connected(x,num_outputs=n_class)

    return z_mu, z_lv, y_predict

    
def Generator(z, reuse = False):
    if (reuse):
        tf.get_variable_scope().reuse_variables()
    # with tf.variable_scope('Generator'):
    with slim.arg_scope(
                        [slim.batch_norm],
                        scale=True,
                        updates_collections=None,
                        decay=0.9, epsilon=1e-5,
                        is_training=True,
                        scope='BN'):
        x = slim.fully_connected(
            z,
            14 * 14 * 256,
            normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.relu)
        x = tf.reshape(x, [-1, 14, 14, 256])
        with slim.arg_scope(
                [slim.conv2d_transpose],
                weights_regularizer=slim.l2_regularizer(1e-6),
                normalizer_fn=slim.batch_norm,
                activation_fn=tf.nn.relu):
            x = slim.conv2d_transpose(x,256, [5,5], [2,2])
            x = slim.conv2d_transpose(x,128, [5,5], [2,2])
            x = slim.conv2d_transpose(x,64, [5,5], [2,2])
            
        #  Don't apply BN for the last layer of G
            x = slim.conv2d_transpose(x,3, [5,5], [2,2],
                                    normalizer_fn=None,
                                    activation_fn=tf.nn.tanh)
    return x
    
    
def Discriminator(x, reuse = False):
    if (reuse):
        tf.get_variable_scope().reuse_variables()
    feature = list()
    with tf.variable_scope('Discriminator'):
        with slim.arg_scope(
                [slim.batch_norm],
                scale=True,
                updates_collections=None,
                decay=0.9, epsilon=1e-5,
                is_training=True,
                # reuse=None
                scope='BN'):
            with slim.arg_scope(
                    [slim.conv2d],
                    weights_regularizer=slim.l2_regularizer(1e-3),
                    normalizer_fn=slim.batch_norm,
                    activation_fn=tf.nn.relu):
                x = slim.conv2d(x,32,[5,5],[2,2],normalizer_fn=None)
                x = slim.conv2d(x,128,[5,5],[2,2])
                x = slim.conv2d(x,256,[5,5],[2,2])
                x = slim.conv2d(x,512,[5,5],[2,2])
        feature.append(x)
        x = slim.flatten(x)
        h = slim.flatten(feature[-1])
        x = slim.fully_connected(
            x,
            1,
            weights_regularizer=slim.l2_regularizer(1e-3),
            activation_fn=None)
    
    return x,h

    
def loss(xs, xt, ys):
    
    def mean_sigmoid_cross_entropy_with_logits(logit, truth):
            '''
            truth: 0. or 1.
            '''
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,
                                                                          labels=truth * tf.ones_like(logit)))
        
    #source domain
    with tf.variable_scope("Encoder"):
        z_mu, z_lv, y_predict = Encoder(xs, z_dim=256,n_class=10)

    z = tools.GaussianSampleLayer(z_mu, z_lv)
    # direct sample
    z_direct = tf.random_normal(shape=tf.shape(z_mu))

    with tf.variable_scope("Generator"):
        xz = Generator(z_direct)
    with tf.variable_scope("Discriminator"):
        logit_fake_xz,_ = Discriminator(xz)
    #
    with tf.variable_scope("Generator", reuse=True):
        xh = Generator(z)
    with tf.variable_scope("Discriminator", reuse=True):
        logit_true, x_through_D = Discriminator(xs, reuse=True)
    with tf.variable_scope("Discriminator", reuse=True):
        logit_fake, xh_through_D = Discriminator(xh, reuse=True)
    
    # target domain
    with tf.variable_scope("Encoder", reuse=True):
        z_mu_t,_, yt_predict = Encoder(xt, z_dim=256,n_class=10)
    with tf.variable_scope("Generator", reuse=True):
        xh_t = Generator(z_mu_t, reuse=True)
    with tf.variable_scope("Discriminator", reuse=True):
        logit_fake_t, _ = Discriminator(xh_t, reuse=True)
    
    loss = dict()
    loss['D_real'] = \
                mean_sigmoid_cross_entropy_with_logits(logit_true, 1.)
    loss['D_fake'] = 0.5 * (
                mean_sigmoid_cross_entropy_with_logits(logit_fake, 0.) +\
                mean_sigmoid_cross_entropy_with_logits(logit_fake_xz, 0.))
    loss['G_fake'] = 0.5 * (
                mean_sigmoid_cross_entropy_with_logits(logit_fake, 1.) +\
                mean_sigmoid_cross_entropy_with_logits(logit_fake_xz, 1.))
    loss['KL(z)'] = tf.reduce_mean(
                    tools.GaussianKLD(
                        z_mu, z_lv,
                        tf.zeros_like(z_mu), tf.zeros_like(z_lv)))
    
    loss['Dis'] = - tf.reduce_mean(
                    tools.GaussianLogDensity(
                        x_through_D,
                        xh_through_D,
                        tf.zeros_like(xh_through_D)))
    ys = tf.cast(ys, tf.float32)
    y_predict= tf.cast(y_predict, tf.float32)
    loss['C_predict'] = tools.loss(ys,y_predict )
    loss['MMD'] = tf.abs(tf.reduce_mean(tf.abs(z_mu-z_mu_t)))
    loss['D_fake_t'] = mean_sigmoid_cross_entropy_with_logits(logit_fake_t, 0.)
    
    return loss, yt_predict
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
