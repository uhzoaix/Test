import numpy as np
import tensorflow as tf
import net

import data_input
from PIL import Image
from datetime import datetime

class MyDNN():
	"""docstring for MyDNN"""
	def __init__(self):
		super(MyDNN, self).__init__()
		self.lr_ = 0.5
		self.batch_size_ = -1
		self.max_iters_ = 10000
		self.display_step_ = 50


	def init_parameters(self, 	learning_rate, 
								batch_size,
								max_iters,
								regularization_lambda,
								):
		self.lr_ = learning_rate
		self.batch_size_ = batch_size
		self.max_iters_ = max_iters
		self.reg_para_ = regularization_lambda

	def get_log_name(self):
		lr = self.lr_
		bs = self.batch_size_
		max_iters = self.max_iters_
		reg_para = self.reg_para_

		return "lr{}_bs{}_maxiters{}_reg{}".format(lr, bs, max_iters, reg_para)

	def get_label_batch(self, origin_label_batch, classes):
		w = origin_label_batch.shape[0]
		# print("Tensor shape: ", origin_label_batch.shape)
		# print("w: {}".format(w))

		result = np.zeros((w,classes))
		for i in range(w):
			j = origin_label_batch[i]
			try:
				result[i,j] = 1
			except IndexError:
				print("i:{}, j:{}".format(i,j))
				raise IndexError


		return result


	def train(self, train_size, test_size, data_path, logname):
		lr = self.lr_
		batch_size = self.batch_size_

		f= open('./log_file/' + logname + '.txt', 'w')
		print("Train log file", file=f)
		print("Time: ", str(datetime.now()), file=f)

		# get images and label batch
		with tf.name_scope("input"):
			train_image_batch, train_label_batch = data_input.get_images_batch_with_labels(
				data_path = data_path + "train/",
				shape=[200,200],
				channels=3,
				batch_size=batch_size,
				data_size=train_size)

			test_image_batch, test_label_batch = data_input.get_images_batch_with_labels(
				data_path = data_path + "test/",
				shape=[200,200],
				channels=3,
				batch_size = batch_size,
				data_size=test_size)


			w, h, c, classes = 200, 200 ,3, 5
			x = tf.placeholder(tf.float32, [batch_size, w, h, c])
			y = tf.placeholder(tf.float32, [batch_size, classes])
		# construct the loss function and the optimizer

		with tf.name_scope("Logits"):
			# pred = net.alexnet(x, classes)
			pred = net.alexnet(x, classes)
			tf.summary.histogram("pred", pred)

		with tf.name_scope("loss"):
			loss, batch_loss, reg_loss = net.loss(_logits=pred, _labels=y, regularization_lambda=self.reg_para_)

		with tf.name_scope("optimization"):
			optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

			grads = tf.gradients(loss, pred)
			tf.summary.histogram("grad", grads)

		with tf.name_scope("accuracy"):
			# Model Evaluation
			correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
			acc_summary = tf.summary.scalar("accuracy", accuracy)

		# for summary issues

		summary_op = tf.summary.merge_all()

		# init variables
		init = tf.global_variables_initializer()

		# saver to save all the variables to checkpoint file
		saver = tf.train.Saver()

		with tf.Session() as sess:
			sess.run(init)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			summary_writer = tf.summary.FileWriter('./tmp/train/' + logname, graph=sess.graph)
			summary_writer_test = tf.summary.FileWriter('./tmp/test/' + logname, graph=sess.graph)

			step = 1
			max_test_acc = -1

			while step <= self.max_iters_:
				image_batch, label_batch = sess.run([train_image_batch, train_label_batch])
				label_batch = self.get_label_batch(label_batch, classes)

				sess.run(optimizer, feed_dict = {x : image_batch, y : label_batch})

				if step % self.display_step_ == 0:
					b_l, reg_l, acc, summary = sess.run ([batch_loss, reg_loss, accuracy, summary_op], feed_dict = {x : image_batch, y: label_batch})

					print("[Step{}]".format(step))
					print("[Step{}]The batch loss is {:.6f}, reg loss is {:.6f}, accuracy is {:.5f}".format(step, b_l, reg_l, acc))

					summary_writer.add_summary(summary, step)

				if step % (10 * self.display_step_) == 0:
					# calculate the accuracy on the test dataset
					print("[STEP{}]test time!".format(step))
					max_steps = int(test_size / batch_size)

					test_acc = 0
					for i in range(max_steps):
						test_images, test_labels = sess.run([test_image_batch, test_label_batch])
						test_labels = self.get_label_batch(test_labels, classes)

						test_batch_acc = sess.run(accuracy, feed_dict = {x : test_images, y : test_labels})

						test_acc += test_batch_acc

					test_acc = test_acc / max_steps
					if test_acc > max_test_acc:
						max_test_acc = test_acc

					print("[STEP{}]The test accuracy is:{}".format(step, test_acc))
					print("[STEP{}]The test accuracy is:{}".format(step, test_acc), file=f)
					print("[STEP{}]test complete!".format(step))

				step += 1

			print("Optimization Finished!")
			
			saver.save(sess, "./model/" + logname + "_model.ckpt")

			coord.request_stop()
			coord.join(threads)
			print("Have save all variables to " + "./model/"+ logname + "_model.ckpt")

		print("Max test accuracy:{}".format(max_test_acc), file=f)
		print("End time: ", str(datetime.now()), file=f)
		f.close()


if __name__ == "__main__":
	# test_x = np.random.rand(8,5,5,3)
	# test_y = np.random.rand(8,4)

	test_nn = MyDNN()

	test_nn.init_parameters(
				learning_rate=0.001, 
				batch_size=50, 
				max_iters=80000, 
				regularization_lambda=0.0005)

	log_name = test_nn.get_log_name()

	with tf.Graph().as_default():
		is_linux = True
		if is_linux:
			test_nn.train(
					train_size = 3600, 
					test_size=300,
					data_path="/home/abaci/uhzoaix/data/",
					logname = 'refined_' + log_name)
		else :
			test_nn.train(
					train_size = 3600, 
					test_size=500, 
					data_path="../data/cat/",
					logdir = './tmp/alexnet')


