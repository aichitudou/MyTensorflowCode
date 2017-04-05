import tensorflow as tf

a = tf.constant([
	[[1.0,2.0,3.0],
	[5.0,6.0,7.0],
	[8.0,7.0,6.0]]
])

a = tf.reshape(a,[1,3,3,1])

pooling = tf.nn.max_pool(a,[1,2,2,1],[1,2,2,1],padding='SAME')

with tf.Session() as sess:
	print("image:")
	image=sess.run(a)
	print (image.shape)
	print (image)
	print("result:")
	result=sess.run(pooling)
	print(result.shape)
	print(result)






