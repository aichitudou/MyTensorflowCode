import os
import tensorflow as tf

fixed_height = 24
fixed_width = 24

train_samples_per_epoch = 50000
test_samples_per_epoch = 10000
data_dir = './cifar-10-batches-bin'
batch_size = 128


def read_cifar10(filename_queue):
	class Image(object):
		pass
	image = Image()
	image.height = 32
	image.width = 32
	image.depth = 3
	label_bytes = 1

	image_bytes = image.height*image.width*image.depth
	Bytes_to_read = label_bytes + image_bytes

	reader = tf.FixedLengthRecordReader(record_bytes=Bytes_to_read)
	
	image.key, value_str = reader.read(filename_queue)
	value = tf.decode_raw(bytes=value_str, out_type=tf.uint8)

	image.label = tf.slice(input=value, begin=[0], size=[label_bytes])
	data_mat = tf.slice(input=value, begin=[label_bytes], size=[image_bytes])
	data_mat = tf.reshape(data_mat, (image.depth, image.height, image.width))
	transposed_value = tf.transpose(data_mat, perm=[1,2,0]) 
	image.mat = transposed_value

	return image

def get_batch_samples(img_obj, min_samples_in_queue, batch_size, shuffle_flag):

	if shuffle_flag == False:
		image_batch, label_batch = tf.train.batch(tensors=img_obj,
													batch_size = batch_size,
													num_threads=4,
													capacity=min_sample_in_queue + 3 * batch_size
													)
	else:
		image_batch, label_batch = tf.train.shuffle_batch(tensors=img
														batch_size = batch_size,
														num_threads = 4,
														min_after_dequeue = min_sample_in_queue,
														capacity = min_sample_in_queue + 3 * batch_size													
														)
	tf.image_summary('input_image', image_batch, max_image=6)

	return image_batch, tf.reshape(label_batch, shape=[batch_size])


def preprocess_input_data(){
	filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1,6)]	
	#filenames = [os.path.join(data_dir, 'test_batch.bin')]
	for f in filenames:
		if not tf.gfile.Exists(f):
			raise ValueError('fail to find file: 'f)
	filename_queue = tf.train.string_input_producer(string_tensor = filenames)
	image = read_cifar10(filename_queue)
	
	new_img = tf.cast(image.mat, tf.float32)
	tf.image_summary('raw_input_image', tf.reshape(new_img, [1,32,32,3]))
	new_img = tf.random_crop(new_img, size=(fixed_height, fixed_width, 3))
	new_img = tf.image.random_brightness(new_img, max_delta=63)
	new_img = tf.image.random_flip_left_right(new_img)
	new_img = tf.image.random_contrast(new_img, lower=0.2, upper=1.8)
	final_img = tf.image.per_image_whitening(new_img)

	min_samples_ratio_in_queue = 0.4 
	min_samples_in_queue = int(min_samples_ratio_in_queue * train_samples_per_epoch)
	
	return get_batch_samples([final_img, image.label], min_samples_in_queue, batch_size, shuffle_flag=True)



	



}

