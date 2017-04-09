# -*- coding: utf-8 -*-

import tensorflow as tf


#保存时dtype类型要一致，一般使用float32，另外要定义变量名
w=tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
b=tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')

#初始化所有变量
init = tf.initialize_all_variables()
#构建保存模型
saver=tf.train.Saver()
#启动
with tf.Session() as sess:
	sess.run(init)
	#定义保存路径
	save_path = saver.save(sess, "save/save_test.ckpt")
	print ("Save to path : ", save_path)



