import numpy as np
import tensorflow as tf
import net

import data_input
from PIL

class MyDNN():
	"""docstring for MyDNN"""
	def __init__(self):
		super(MyDNN, self).__init__()
		self.lr_ = 0.5
		self.batch_size_ = -1
		self.images_num = -1
		self.max_iters_ = 10000
		self.display_step_ = 50


	def init_parameters(self, 	learning_rate, 
								batch_size,
								images_num,
								max_iters
								):
		self.lr_ = learning_rate
		self.batch_size_ = batch_size
		self.images_num_ = images_num
		self.max_iters_ = max_iters


	def get_batch_range(self, step):
		bs = self.batch_size_
		images_num = self.images_num_
		if images_num < bs:
			raise ValueError

		max_steps = int(images_num / bs)
		step = step % max_steps if step % max_steps > 0 else max_steps
		return range((step - 1) * bs, step * bs)

	def get_label_batch(self, origin_label_batch, classes):
		w = origin_label_batch.shape[0]
		# print("Tensor shape: ", origin_label_batch.shape)
		# print("w: {}".format(w))

		result = np.zeros((w,classes))
		for i in range(w):
			j = origin_label_batch[i]
			result[i,j] = 1

		return result

	def train(self, images_num, data_path):
		lr = self.lr_
		batch_size = self.batch_size_

		# get images and label batch
		train_image_batch, train_label_batch, _, _ = data_input.get_images_batch_with_labels(
			data_path=data_path,
			shape=[200,200],
			channels=3,
			batch_size=batch_size,
			train_size=images_num,
			test_size=0)

		w, h, c, classes = 200, 200 ,3, 5
		x = tf.placeholder(tf.float32, [batch_size, w, h, c])
		y = tf.placeholder(tf.float32, [batch_size, classes])
		print("X: ", x.name)
		print("Y: ", y.name)
		# construct the loss function and the optimizer

		with tf.name_scope("Logits"):
			pred = net.network(x)

		with tf.name_scope("loss"):
			loss = net.loss(_logits=pred, _labels=y)

		with tf.name_scope("optimization"):
			optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

		with tf.name_scope("accuracy"):
			# Model Evaluation
			correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		# for summary issues
		tf.summary.scalar("loss", loss)
		tf.summary.scalar("accuracy", accuracy)

		summary_op = tf.summary.merge_all()

		# init variables
		init = tf.global_variables_initializer()

		# saver to save all the variables to checkpoint file
		saver = tf.train.Saver()

		with tf.Session() as sess:
			sess.run(init)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			summary_writer = tf.summary.FileWriter("./tmp/train/", graph=sess.graph)

			step = 1

			while step <= self.max_iters_:
				image_batch, label_batch = sess.run([train_image_batch, train_label_batch])
				label_batch = self.get_label_batch(label_batch, classes)

				sess.run(optimizer, feed_dict = {x : image_batch, y : label_batch})

				if step % self.display_step_ == 0:
					l, acc, summary = sess.run ([loss, accuracy, summary_op], feed_dict = {x : image_batch, y: label_batch})

					print("[Step{}]".format(step))
					print("Now, the batch loss is {:.6f}, accuracy is {:.5f}".format(l, acc))

					summary_writer.add_summary(summary, step)

				step += 1

			print("Optimization Finished!")
			
			saver.save(sess, "./tmp/model.ckpt")

			coord.request_stop()
			coord.join(threads)
			print("Have save all variables to " + "./tmp/model.ckpt")


if __name__ == "__main__":
	# test_x = np.random.rand(8,5,5,3)
	# test_y = np.random.rand(8,4)

	test_nn = MyDNN()
	# test_nn.init_parameters(learning_rate=0.5, batch_size=2)
	# test_nn.train(test_x, test_y)
	test_nn.init_parameters(learning_rate=0.1, batch_size=20, images_num=8, max_iters=10000)

	with tf.Graph().as_default():
		is_linux = False
		if is_linux:
			test_nn.train(images_num = 3600, data_path="/home/abaci/uhzoaix/cat/")
		else :
			test_nn.train(images_num=3600, data_path="../data/cat/")

