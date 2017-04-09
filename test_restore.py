# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

#重新定义相同的变量的dtype和shape
w = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name='weights')
b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases')

change = tf.reshape(w, [3,2])

#不需要初始化

saver=tf.train.Saver()
with tf.Session() as sess:
	saver.restore(sess,"save/save_test.ckpt")
	print("weights:", sess.run(change))
	print("biases:", sess.run(b))



