import tensorflow as tf


weights = {
	'wd1': tf.Variable(tf.random_normal([4*4*256, 1024]))

}

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)

	print (weights['wd1'].get_shape().as_list())





