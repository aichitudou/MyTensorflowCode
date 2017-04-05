import tensorflow as tf
import input_dataset

height = input_dataset.fixed_height
width = input_dataset.fixed_width

train_samples_per_epoch = input_dataset.train_samples_per_epoch
test_samples_per_epoch = input_dataset.test_samples_per_epoch

moving_average_decay = 0.9999
num_epochs_per_decay = 350
learning_rate_decay_factor = 0.1
initial_learning_rate = 0.1


def variable_on_cpu(name, shape, dtype, initializer):
	with tf.device("/cpu:0"):
		return tf.get_variable(name=name, 
								shape=shape,
								initializer=initializer,
								dtype=dtype)


def variable_on_cpu_with_collection(name, shape, dtype, stddev, wd):
	with tf.device("/cpu:0"):
		weight = tf.get_variable(name=name,
								shape=shape,
								initializer=tf.truncated_normal_initializer(stddev=stddev,dtype=dtype))
		if wd is not None:
				weight_decay = tf.mul(tf.nn.l2_loss(weight), wd, name='weight_loss')
				tf.add_to_collection(name='losses', value=weight_decay)
		return weight

def losses_summary(total_loss):
	average_op = tf.train.ExponentialMovingAverage(decay=0.9)
	losses=tf.get_collection(key='losses')

	maintain_averages_op = average_op.apply(losses+[total_loss])
	for i in losses+[total_loss]:
		tf.scalar_summary(i.op.name+'_raw', i)
		tf.scalar_summary(i.op.name, average_op.average(i))
	return maintain_average_op


def one_step_train(total_loss, step):
	batch_cout = int(train_sample_per_epoch / input_dataset.batch_size)
	decay_step = batch_count * num_epochs_per_decay 
	lr = tf.train.exponential_decay(learning_rate = initial_learning_rate,
									global_step = step,
									decay_steps = decay_step,
									decay_rate = learning_rate_decay_factor,
									staircase = True)
	tf.scalar_summary('learning_rate', lr)
	losses_movingaverage_op = losses_summary(total_loss)
	
	with tf.control_dependencies(control_inputs=[losses_movingaverage_op]):
		trainer=tf.train.GradientDescentOptimizer(learning_rate=lr)
		gradient_pairs = trainer.compute_gradients(loss=total_loss)
	gradient_update = trainer.apply_gradients(grads_and_vars=gradient_pairs, global_step=step)
	variables_average_op = tf.train.ExponentialMovingAverage(decay=moving_average_decay, num_updates=step)
	
	maintain_variable_average_op = variable_average_op.apply(var_list=tf.trainable_variable())
	with tf.control_dependencies(control_inputs=[gradient_update, maintain_variable_average_op]):
		gradient_update_optimizer = tf.no_op()
	
	return gradient_update_optimizer
	




def network(images):
	with tf.variable_scope(name_or_scope='conv1') as scope:
		weight = variable_on_cpu_with_collection(name='weight',
												shape=(5,5,3,64),
												dtype=tf.float32,
												stddev=0.05,
												wd=0.0
												)
		bias = variable_on_cpu(name='bias', shape(64),dtype=tf.float32, 
								initializer=tf.constant_initializer(value=0.0))
		conv1_in = tf.nn.conv2d(input=images, filter=weight, strides=(1,1,1,1),padding='SAME')
		conv1_in = tf.nn.bias_add(value=conv1_in, bias=bias)
		conv1_out = tf.nn.relu(conv1_in)

	pool1 = tf.nn.max_pool(value=conv1_out, ksize=(1,3,3,1), strides=(1,2,2,1), padding='SAME')

	norm1 = tf.nn.lrn(input=pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75)

	with tf.variable_scop(name_or_scope='conv2') as scope:
		weight = variable_on_cpu_with_collection(name='weight',
												shape=(5,5,64,64),
												dtype=tf.float32,
												stddev=0.05,
												wd=0.0)
		bias = variable_on_cpu(name='bias', shape=(64), dtype=tf.float32, 
								initializer=tf.constant_initializer(value=0.1))
		conv2_in = tf.nn.conv2d(norm1, weight, stides=(1,1,1,1), padding='SAME')
		conv2_in = tf.nn.bias_add(conv2_in, bias)
		conv2_out = tf.nn.relu(conv2_in)

	norm2 = tf.nn.lrn(input=conv2_out, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75)

	pool2 = tf.nn.max_pool(value=norm2, ksize=(1,3,3,1), strides=(1,2,2,1),padding='SAME')

	# input tensor of shape '[batch, in_height, in_width, in_channels]'
	reshaped_pool2 = tf.reshape(tensor=pool2, shape=(-1, 6*6*64))

	with tf.variable_scope(name_or_scope='fully_connected_layer1') as scope:
		weight = variable_on_cpu_with_collection(name='weight', 
												shape=(6*6*64, 384),
												dtype=tf.float32,
												stddev=0.04,
												wd == 0.004)
		bias = variable_on_cpu(name='bias', shape=(384), dtype=tf.float32, 
								initializer=tf.constant_initializer(value=0.1))
		fc1_in = tf.matmul(reshaped_pool2, weight) + bias
		fc1_out = tf.nn.relu(fc1_in)

	with tf.variable_scope(name_or_scope='fully_connected_layer2') as scope:
		weight = variable_on_cpu_with_collection(name='weight',
												shape=(384, 192),
												dtype=tf.float32, 
												stddev=0.04,
												wd=0.004)
		bias = variable_on_cpu(name='bias', shape=(192), dtype=tf.float32, 
								initializer=tf.constant_initializer(value=0.1))
		fc2_in = tf.matmul(fc1_out, weight) + bias
		fc2_out = tf.nn.relu(fc2_in)

	with tf.variable_scope(name_or_scope='softmax_layer') as scope:
		weight = variable_on_cpu_with_collection(name='weight',
												shape=(192,10),
												dtype=tf.float32,
												stddev=1/192,
												wd=0.0)
		bias = variable_on_cpu(name='bias', shape=(10), dtype=tf.float32, 
								initializer = tf.constant_initializer(value=0.0))

		classifier_in = tf.matmul(fc2_out, weight) + bias
		classifier_out = tf.nn.softmax(classifier_in)
	
	return classifier_in

		
def loss(logits, labels):
	labels = tf.cast(x=labels, dtype=tf.int32)
	cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label,
																			name = 'likelihood_loss')
	cross_entropy_loss = tf.reduce_mean(cross_entropy_loss, name='cross_entropy_loss')
	
	tf.add_to_collection(name='losses', value=cross_entropy_loss)
	return tf.add_n(inputs=tf.get_collection(key='losses'), name='total_loss')


