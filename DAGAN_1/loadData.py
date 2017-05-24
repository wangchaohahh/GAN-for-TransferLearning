"""
this is a file to load small images and labels
@author: wangchao
@date : 2016.12
"""

import os
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from scipy.misc import imread, imresize
import numpy as np
from sklearn.utils import shuffle
from sklearn import manifold
from time import time



def loaddata(filePath):
    images = []
    labels = []
    i = 0
    ten_class = ['projector','bike','calculator','keyboard','mug',
                 'headphones','monitor','laptop_computer','back_pack','mouse']
    # for datapath in os.listdir(filePath):
    for datapath in ten_class:
        for filename in os.listdir(filePath + '/' + datapath):
            filename = filePath + '/' + datapath + '/' + filename
            img = imread(filename)
            img = imresize(img, (28, 28))
            labels.append(i)
            images.append(img)
        i += 1
    images = np.asarray(images)
    labels = np.array(labels).astype(float)
    images, labels = shuffle(images, labels, random_state=0)
    return images, labels



# test load data
# images, labels = loaddata("/home/wangchao/deeplearning/transfer_learning/dataset/webcam/images")
# np.savetxt('test.tsv',delimiter='\t')


# tensor_images = tf.placeholder('uint8',shape=images.shape)
# batch_img = tf.train.shuffle_batch([tensor_images],30,1000,500,enqueue_many= True)
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     batch_img = sess.run(batch_img,feed_dict={tensor_images:images})
#     print batch_img


# visualize
images, labels = loaddata("/home/wangchao/deeplearning/transfer_learning/dataset/webcam/images")
n_sample = images.shape[0]
images = np.reshape(images,(n_sample,-1))

n_feature = images.shape[1]
n_neighbors = 30

