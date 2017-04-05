import tensorflow as tf
import input_dataset
import forward_prop
import train
import math
import numpy as np


def eval_once(summary_op, summary_writer, saver, predict_true_or_false):
	with tf.Session() as sess:
		checkpoint_proto = tf.train.get_checkpoint_state(checkpoint_dir=train.checkpoint_path)
		if checkpoint_proto and checkpoint_proto.model_checkpoint_path:
			saver.restore(sess, checkpoint_proto.model_checkpoint_path)
		else:
			print('checkpoint file not found')
			return
		coord = tf.train.Coordinate()

		try:
			threads = []
			for queue_runner in tf.get_collection(key=tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(queue_runner.create_threads(sess, coord=coord, daemon=True, start=True))
			test_batch_num = math.ceil(input_dataset.test_sample_per_epoch/input_dataset.batch_size)
			iter_num = 0
			true_test_num = 0
			
			total_test_num = test_batch_num * input_dataset.batch_size

			while iter_num < test_batch_num and not coord.should_stop():
				result_judge = sess.run([predict_true_or_false])
				true_test_num += np.sum(result_judge)
				iter_num += 1

			precision = true_test_num / total_test_num
			print("The test precision is %.3f" % precision)

		except:
			coord.request_stop()
		coord.request_stop()
		coord.join(threads)
			





def evaluate():
	with tf.Graph().as_default() as g:
		img_batch, labels = input_dataset.input_data(eval_flag=True)
		logits = forward_prop.network(img_batch)

		predict_true_or_false = tf.nn.in_top_k(predictions=logits, targets=labels, k=1)

		moving_average_op = tf.trian.ExponentialMovingAverage(decay=forward_prop, moving_average_decay)
		variables_to_restore = moving_average_op.variables_to_restore()
		saver = tf.train.Saver(var_list=variables_to_restore)

		summary_op = tf.merge_all_summaries()
		summary_writer = tf.train.SummaryWriter(logdir='./event-log-test', graph=g)
		eval_once(summary_op, summary_writer, saver, predict_true_or_false)
