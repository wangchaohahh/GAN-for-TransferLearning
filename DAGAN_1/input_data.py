# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 13:23:41 2017

@author: wangchao
"""

import tensorflow as tf
import matplotlib.pyplot as plt

## load source and target domain data

def read_and_decode(filename, batch_size, is_batch = True):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    
    # data argumentation
    image = tf.random_crop(img, [224, 224, 3])# randomly crop the image size to 224 x 224
    # image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_brightness(image, max_delta=63)
    # image = tf.image.random_contrast(image,lower=0.2,upper=1.8)
    
    img = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    if is_batch:
        image, label_batch = tf.train.shuffle_batch(
                                        [img, label],
                                        batch_size = batch_size,
                                        num_threads= 16,
                                        capacity = 20000,
                                        min_after_dequeue = 3000)
        ## ONE-HOT
        n_classes = 10
        label_batch = tf.one_hot(label_batch, depth= n_classes)
        label_batch = tf.cast(label_batch, dtype=tf.int32)
        label_batch = tf.reshape(label_batch, [batch_size, n_classes])
    else:
        image = img
        label_batch = label
        n_classes = 10
        label_batch = tf.one_hot(label_batch, depth=n_classes)
        label_batch = tf.cast(label_batch, dtype=tf.int32)
        label_batch = tf.reshape(label_batch, [-1, n_classes])
    return image, label_batch
        
## test

# img, labels = read_and_decode('dslr.tfrecords',64)
# img_test = tf.reshape(img, [64,224,224,3])
# init = tf.global_variables_initializer()
# # with tf.Session() as sess:
# #     sess.run(init)
# #     threads = tf.train.start_queue_runners(sess=sess)
# #     val, l= sess.run([img, labels])
# #     print (val.shape, l.shape)
#
# with tf.Session()  as sess:
#
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     try:
#         for i in range(3):
#             # just plot one batch size
#             # image, label = sess.run([img, labels])
#             image = sess.run([img_test])
#             print(image)
#             # plt.imshow(image[10])
#             # plt.show()
#
#
#     except tf.errors.OutOfRangeError:
#         print('done!')
#     finally:
#         coord.request_stop()
#         coord.join(threads)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
