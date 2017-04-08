# -*- coding: utf-8 -*-
import tensorflow as tf
from input_data import encode_to_tfrecords,decode_from_tfrecords,get_batch,get_test_batch
import cv2
import os

import sys
default_encoding='utf-8'
if sys.getdefaultencoding()!=default_encoding:
	reload(sys)
	sys.setdefaultencoding(default_encoding)


class network(object):
	def __init__(self):
		#初始化权值和偏置
		with tf.variable_scope("weights"):
			self.weights={
				'conv1':tf.get_variable('conv1',[11,11,3,96],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
				'conv2':tf.get_variable('conv2',[5,5,96,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
				'conv3':tf.get_variable('conv3',[3,3,256,384],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
				'conv4':tf.get_variable('conv4',[3,3,384,384],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
				'conv5':tf.get_variable('conv5',[3,3,384,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
				'fc1':tf.get_variable('fc1',[6*6*256,4096],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
				'fc2':tf.get_variable('fc2',[4096,4096],initializer=tf.contrib.layers.xavier_initializer_conv2d()),

				'fc3':tf.get_variable('fc3',[4096,2],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
			
			}

		with tf.variable_scope("biases"):
			self.biases={
				'conv1':tf.get_variable('conv1',[96,],initializer=tf.constant_initializer(value=0.0,dtype=tf.float32)),
				'conv2':tf.get_variable('conv2',[256,],initializer=tf.constant_initializer(value=0.0,dtype=tf.float32)),
				'conv3':tf.get_variable('conv3',[384,],initializer=tf.constant_initializer(value=0.0,dtype=tf.float32)),
				'conv4':tf.get_variable('conv4',[384,],initializer=tf.constant_initializer(value=0.0,dtype=tf.float32)),
				'conv5':tf.get_variable('conv5',[256,],initializer=tf.constant_initializer(value=0.0,dtype=tf.float32)),

				'fc1':tf.get_variable('fc1',[4096,],initializer=tf.constant_initializer(value=0.0,dtype=tf.float32)),
				'fc2':tf.get_variable('fc2',[4096,],initializer=tf.constant_initializer(value=0.0,dtype=tf.float32)),
				'fc3':tf.get_variable('fc3',[2,],initializer=tf.constant_initializer(value=0.0,dtype=tf.float32))

			}

	def inference(self,images):
		#向量转为矩阵
		images = tf.reshape(images, shape=[-1,227,227,3]) # [batch, in_height, in_width, in_channels]
		images = (tf.cast(images,tf.float32)/255.-0.5)*2 #归一化处理
		
		#第一层 定义卷积偏置和下采样
		conv1=tf.nn.bias_add(tf.nn.conv2d(images,self.weights['conv1'],strides=[1,4,4,1],padding='VALID'),self.biases['conv1'])
		
		relu1=tf.nn.relu(conv1)
		pool1=tf.nn.max_pool(relu1,ksize=[1,3,3,1], strides=[1,2,2,1],padding='VALID')


		#第二层
		conv2=tf.nn.bias_add(tf.nn.conv2d(pool1,self.weights['conv2'],strides=[1,1,1,1],padding='SAME'),self.biases['conv2'])
		relu2=tf.nn.relu(conv2)
		pool2=tf.nn.max_pool(relu2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')

		
		#第三层
		conv3=tf.nn.bias_add(tf.nn.conv2d(pool2,self.weights['conv3'],strides=[1,1,1,1],padding='SAME'),self.biases['conv3'])
		relu3=tf.nn.relu(conv3)
		conv4=tf.nn.bias_add(tf.nn.conv2d(relu3,self.weights['conv4'],strides=[1,1,1,1],padding='SAME'),self.biases['conv4'])
		relu4=tf.nn.relu(conv4)
		conv5=tf.nn.bias_add(tf.nn.conv2d(relu4,self.weights['conv5'],strides=[1,1,1,1],padding='SAME'),self.biases['conv5'])
		relu5=tf.nn.relu(conv5)
		pool5=tf.nn.max_pool(relu5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')

		#全连接层1，先把特征图转为向量
		flatten=tf.reshape(pool5,[-1,self.weights['fc1'].get_shape().as_list()[0]])
		
		drop1=tf.nn.dropout(flatten,0.5)
		fc1=tf.matmul(drop1,self.weights['fc1'])+self.biases['fc1']

		fc_relu1=tf.nn.relu(fc1)

		fc2=tf.matmul(fc_relu1,self.weights['fc2'])+self.biases['fc2']
		fc_relu2=tf.nn.relu(fc2)

		fc3=tf.matmul(fc_relu2,self.weights['fc3'])+self.biases['fc3']

		return fc3


	#计算softmax交叉熵损失函数
	def softmax_entropy_loss(self,predicts,labels):
		predicts=tf.nn.softmax(predicts)
		labels=tf.one_hot(labels,self.weights['fc3'].get_shape().as_list()[1])
		loss=tf.nn.softmax_cross_entropy_with_logits(logits=predicts,labels=labels)#loss=-tf.reduce_mean(labels*tf.log(predicts))
		self.cost=tf.reduce_mean(loss)

		return self.cost

	#梯度下降
	def optimize(self,loss,lr=0.001):
		train_optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
		return train_optimizer
	

	


def train():
	image, label = decode_from_tfrecords('data/train.tfrecords')
	batch_image,batch_label = get_batch(image,label,96,227)#生成测试的样例batch

	#网络连接，训练所有
	net=network()
	inf=net.inference(batch_image)
	loss=net.softmax_entropy_loss(inf,batch_label)
	opti=net.optimize(loss)

	#验证集所用
	test_image,test_label=decode_from_tfrecords('data/val.tfrecords',num_epoch=None)
	test_images,test_labels=get_test_batch(test_image,test_label,batch_size=96,crop_size=227)

	test_inf=net.inference(test_images)
	correct_prediction=tf.equal(tf.cast(tf.argmax(test_inf,1),tf.int32),test_labels)
	accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))



	init=tf.global_variables_initializer()

	with tf.Session() as session:
		session.run(init)
		coord=tf.train.Coordinator()
		threads=tf.train.start_queue_runners(coord=coord)
		max_iter=150
		

		if os.path.exists(os.path.join("model",'model.ckpt')) is True:
			tf.train.Saver(max_to_keep=None).restore(session,os.path.join("model",'model.ckpt'))
		for l in range(max_iter):

			loss_np, _, label_np, image_np = session.run([loss,opti,batch_label,batch_image])
	#		image_np = session.run([batch_image])
			print image_np.shape

			if l%5==0:
				print "Iteration: " + str(iter) + "; Train loss: ", loss_np
			if l%20==0:
				accuracy_np=session.run([accuracy])
				print '***************test accruacy:',accuracy_np,'*******************'
			

		tf.train.Saver(max_to_keep=None).save(session,os.path.join('model','model.ckpt'))
			
		coord.request_stop()#queue需要关闭，否则报错
		coord.join(threads)

def test():
	image, label=decode_from_tfrecords('data/train.tfrecords')
	batch_image, batch_label=get_batch(image,label,96,227)
	
	net=network()
	inf=net.inference(batch_image)
	loss=net.softmax_entropy_loss(inf,batch_label)
	opti=net.optimize(loss,lr=0.001)
	
	#验证
	test_image,test_label=decode_from_tfrecords('data/val.tfrecords')
	test_images,test_labels=get_test_batch(test_image,test_label,batch_size=96,crop_size=227)
	test_inf=net.inference(test_images)
	correct_prediction=tf.equal(tf.cast(tf.argmax(test_inf,1),tf.int32),test_labels)
	accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


	
	init=tf.global_variables_initializer()

	with tf.Session() as session:
		session.run(init)
		coord=tf.train.Coordinator()
		threads=tf.train.start_queue_runners(coord=coord)
		
		if os.path.exists(os.path.join("model",'model.ckpt')) is True:
			tf.train.Saver(max_to_keep=None).restore(session,os.path.join("model",'model.ckpt'))

		for l in range(200):
			_,batch_image_np= session.run([opti,batch_image])

			if l%1 == 0:
				loss_np = session.run(loss)
				print "Iteration:"+str(l)+";Train loss: ",loss_np
				accuracy_np=session.run(accuracy)
				print "Accuracy: ", accuracy_np

			print batch_image_np.shape
		
		tf.train.Saver(max_to_keep=None).save(session,os.path.join('model','model.ckpt'))
		coord.request_stop()
		coord.join(threads)

if __name__=='__main__':
	#train()
	test()
