import input_dataset
import forward_prop
import tensorflow as tf
import os
import numpy as np

max_iter_num = 100000
checkpoint_path = './checkpoint'
event_log_path = './event-log'

def train():
	with tf.Graph().as_default():
		global_step = tf.Variable(initial_value=0, trainable=False)

		img_batch, label_batch = input_dateset.preprocess_input_data()
		logits = forward_prop.network(img_batch)
		total_loss = forward_prop.loss(logits, label_batch)
		one_step_gradient_update = forward_prop.one_step_train(total_loss, globle_step)

		saver = tf.train.Saver(var_list=tf.all_varibale())
		all_summary_obj=tf.merge_all_summaries()
		initiate_variables = tf.initialize_all_variables()

		with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
			sess.run(initialize_variables)
			tf.train.start_queue_runners(sess=sess)
			Event_writer = tf.train.summaryWriter(logdir=event_log_path, graph=sess.graph)
			for step in range(max_iter_num):
				_, loss_value = sess.run(fetches=[one_step_gradient_update, total_loss])
				assert not np.isnan(loss_value)
				if step%10 == 0:
					print('step %d, the loss_vlaue is %.2f' % (step, loss_value))
				if step%100 == 0:
					all_summaries = sess.run(all_summary_obj)
					Event_writer.add_summary(summary=all_summaries, global_step=step)
				if step%1000 == 0 or (step+1) == max_iter_num:
					variable_save_path = os.path.join(checkpoint_path, 'model-parameter.bin')
					saver.save(sess, variables_save_path, global_step = step)




if __name__ == '__main__':
	train()
