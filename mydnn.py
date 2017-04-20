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
								max_iters
								):
		self.lr_ = learning_rate
		self.batch_size_ = batch_size
		self.max_iters_ = max_iters


	def calc_test_accuracy(self, session, test_images, test_labels):


		return acc

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


	def train(self, train_size, test_size, data_path, logdir):
		lr = self.lr_
		batch_size = self.batch_size_

		f= open("log.txt", 'w')
		print("Train log file", file=f)
		print("Time: ", str(datetime.now()), file=f)

		# get images and label batch
		with tf.name_scope("input"):
			train_image_batch, train_label_batch = data_input.get_images_batch_with_labels(
				data_path=data_path + "train/",
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
			pred = net.alexnet(x, classes)
			tf.summary.histogram("pred", pred)

		with tf.name_scope("loss"):
			loss = net.loss(_logits=pred, _labels=y, regularization_lambda=0)
			tf.summary.scalar("loss", loss)

		with tf.name_scope("optimization"):
			optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

		with tf.name_scope("accuracy"):
			# Model Evaluation
			correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
			tf.summary.scalar("accuracy", accuracy)

		# for summary issues
		tf.summary.image("batch images", train_image_batch, max_outputs=20)

		summary_op = tf.summary.merge_all()

		# init variables
		init = tf.global_variables_initializer()

		# saver to save all the variables to checkpoint file
		saver = tf.train.Saver()

		with tf.Session() as sess:
			sess.run(init)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)

			step = 1

			while step <= self.max_iters_:
				image_batch, label_batch = sess.run([train_image_batch, train_label_batch])
				label_batch = self.get_label_batch(label_batch, classes)

				sess.run(optimizer, feed_dict = {x : image_batch, y : label_batch})

				if step % self.display_step_ == 0:
					l, acc, summary = sess.run ([loss, accuracy, summary_op], feed_dict = {x : image_batch, y: label_batch})

					print("[Step{}]".format(step))
					print("[Step{}]The batch loss is {:.6f}, accuracy is {:.5f}".format(step, l, acc))

					summary_writer.add_summary(summary, step)

				if step % (10 * self.display_step_) == 0:
					# calculate the accuracy on the test dataset
					print("[STEP{}]test time!".format(step))
					max_steps = int(test_size / batch_size)

					test_acc = 0
					for i in range(max_steps):
						test_images, test_labels = sess.run([test_image_batch, test_label_batch])
						test_labels = self.get_label_batch(test_labels, classes)

						acc = sess.run(accuracy, feed_dict={x: test_images, y: test_labels})
						test_acc += acc

					print("[STEP{}]The test accuracy is:{}".format(step, test_acc/max_steps))
					print("[STEP{}]The test accuracy is:{}".format(step, test_acc/max_steps), file=f)
					print("[STEP{}]test complete!".format(step))

				step += 1

			print("Optimization Finished!")
			
			saver.save(sess, "./tmp/model.ckpt")

			coord.request_stop()
			coord.join(threads)
			print("Have save all variables to " + "./tmp/model.ckpt")

		f.close()


if __name__ == "__main__":
	# test_x = np.random.rand(8,5,5,3)
	# test_y = np.random.rand(8,4)

	test_nn = MyDNN()
	# test_nn.init_parameters(learning_rate=0.5, batch_size=2)
	# test_nn.train(test_x, test_y)
	test_nn.init_parameters(learning_rate=0.01, batch_size=40, max_iters=100000)

	with tf.Graph().as_default():
		is_linux = True
		if is_linux:
			test_nn.train(
					train_size = 3600, 
					test_size=500, 
					data_path="/home/abaci/uhzoaix/data/",
					logdir = './tmp/alexnet')
		else :
			test_nn.train(
					train_size = 3600, 
					test_size=500, 
					data_path="../data/cat/",
					logdir = './tmp/alexnet')


