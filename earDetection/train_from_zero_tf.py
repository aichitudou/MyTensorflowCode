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
				'conv1':tf.get_variable('conv1',[11,11,3,96],initializer=tf.random_normal_initializer()),
				'conv2':tf.get_variable('conv2',[5,5,96,256],initializer=tf.random_normal_initializer()),
				'conv3':tf.get_variable('conv3',[3,3,256,384],initializer=tf.random_normal_initializer()),
				'conv4':tf.get_variable('conv4',[3,3,384,384],initializer=tf.random_normal_initializer()),
				'conv5':tf.get_variable('conv5',[3,3,384,256],initializer=tf.random_normal_initializer()),
				'fc1':tf.get_variable('fc1',[6*6*256,4096],initializer=tf.random_normal_initializer()),
				'fc2':tf.get_variable('fc2',[4096,4096],initializer=tf.random_normal_initializer()),

				'fc3':tf.get_variable('fc3',[4096,2],initializer=tf.random_normal_initializer()),
			
			}

		with tf.variable_scope("biases"):
			self.biases={
				'conv1':tf.get_variable('conv1',[96,],initializer=tf.constant_initializer(value=1.0,dtype=tf.float32)),
				'conv2':tf.get_variable('conv2',[256,],initializer=tf.constant_initializer(value=1.0,dtype=tf.float32)),
				'conv3':tf.get_variable('conv3',[384,],initializer=tf.constant_initializer(value=1.0,dtype=tf.float32)),
				'conv4':tf.get_variable('conv4',[384,],initializer=tf.constant_initializer(value=1.0,dtype=tf.float32)),
				'conv5':tf.get_variable('conv5',[256,],initializer=tf.constant_initializer(value=1.0,dtype=tf.float32)),

				'fc1':tf.get_variable('fc1',[4096,],initializer=tf.constant_initializer(value=1.0,dtype=tf.float32)),
				'fc2':tf.get_variable('fc2',[4096,],initializer=tf.constant_initializer(value=1.0,dtype=tf.float32)),
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
		norm1=tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9,beta=0.75)
		norm1=tf.nn.dropout(norm1,0.8)


		#第二层
		conv2=tf.nn.bias_add(tf.nn.conv2d(norm1,self.weights['conv2'],strides=[1,1,1,1],padding='SAME'),self.biases['conv2'])
		relu2=tf.nn.relu(conv2)
		pool2=tf.nn.max_pool(relu2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
		norm2=tf.nn.lrn(pool2,4,bias=1.0,alpha=0.001/9,beta=0.75)
		norm2=tf.nn.dropout(norm2,0.8)	
		#第三层
		conv3=tf.nn.bias_add(tf.nn.conv2d(norm2,self.weights['conv3'],strides=[1,1,1,1],padding='SAME'),self.biases['conv3'])
		relu3=tf.nn.relu(conv3)
		conv4=tf.nn.bias_add(tf.nn.conv2d(relu3,self.weights['conv4'],strides=[1,1,1,1],padding='SAME'),self.biases['conv4'])
		relu4=tf.nn.relu(conv4)
		conv5=tf.nn.bias_add(tf.nn.conv2d(relu4,self.weights['conv5'],strides=[1,1,1,1],padding='SAME'),self.biases['conv5'])
		relu5=tf.nn.relu(conv5)
		pool5=tf.nn.max_pool(relu5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')

		#全连接层1，先把特征图转为向量
		flatten=tf.reshape(pool5,[-1,self.weights['fc1'].get_shape().as_list()[0]])
		fc1=tf.matmul(flatten,self.weights['fc1'])+self.biases['fc1']
		fc_relu1=tf.nn.relu(fc1)
		drop1=tf.nn.dropout(fc_relu1,0.8)

		fc2=tf.matmul(fc_relu1,self.weights['fc2'])+self.biases['fc2']
		fc_relu2=tf.nn.relu(fc2)
		drop2=tf.nn.dropout(fc_relu2,0.8)
		fc3=tf.matmul(drop2,self.weights['fc3'])+self.biases['fc3']

		return fc3


	#计算softmax交叉熵损失函数
	def softmax_entropy_loss(self,predicts,labels):
		predicts=tf.nn.softmax(predicts)
		labels=tf.one_hot(labels,self.weights['fc3'].get_shape().as_list()[1])
		loss=tf.nn.softmax_cross_entropy_with_logits(logits=predicts,labels=labels)#loss=-tf.reduce_mean(labels*tf.log(predicts))
		self.cost=tf.reduce_mean(loss)

		return self.cost

	#梯度下降
	def optimize(self,loss,lr=0.0001):
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

	global_step=tf.Variable(0,trainable=False)
	image, label=decode_from_tfrecords('data/train.tfrecords')
	batch_image, batch_label=get_batch(image,label,96,227)
	
	net=network()
	inf=net.inference(batch_image)
	loss=net.softmax_entropy_loss(inf,batch_label)
	learning_rate=tf.train.exponential_decay(
		0.001, #初始学习率
		global_step,#用于衰减计算
		10, #decay_step，每10次，lr衰减一次
		0.95, #decay_rate
		staircase=True
	)
#	opti=net.optimize(loss,lr=0.00001)
	opti=tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=global_step)
		
	#验证
	test_image,test_label=decode_from_tfrecords('data/val.tfrecords')
	test_images,test_labels=get_test_batch(test_image,test_label,batch_size=500,crop_size=227)
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
			_,loss_np = session.run([opti,loss])
			print "Iteration:"+str(l)+";Train loss: ",loss_np
				
			if l%5 == 0:
				accuracy_np=session.run(accuracy)
				print "Accuracy: ", accuracy_np

		
		tf.train.Saver(max_to_keep=None).save(session,os.path.join('model','model.ckpt'))
		coord.request_stop()
		coord.join(threads)

if __name__=='__main__':
	#train()
	test()
