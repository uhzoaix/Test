import numpy as np
import tensorflow as tf
import net

class MyDNN():
	"""docstring for MyDNN"""
	def __init__(self):
		super(MyDNN, self).__init__()
		self.lr_ = 0.5
		self.batch_size_ = -1
		self.images_num = -1
		self.max_iters_ = 10000


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


	def train(self, images, labels):
		lr = self.lr_
		batch_size = self.batch_size_

		if type(images) == np.ndarray:
			# print("Numpy input")
			image_num, w, h, c = images.shape
			label_num, classes = labels.shape	
		else :
			# tensor input
			image_num, w, h, c = images.get_shape().as_list()
			label_num, classes = labels.get_shape().as_list()

		if image_num != self.images_num_:
			print("Wrong with the numbers of images,wtf!?")

		x = tf.placeholder(tf.float32, [batch_size, w, h, c])
		y = tf.placeholder(tf.float32, [batch_size, classes])

		# construct the loss function and the optimizer
		pred = net.network(x)
		loss = net.loss(_logits=pred, _labels=y)

		optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

		# Model Evaluation
		correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		init = tf.global_variables_initializer()

		# saver to save all the variables to checkpoint file
		saver = tf.train.Saver()

		with tf.Session() as sess:
			sess.run(init)

			step = 1
			batch_range = self.get_batch_range(step)
			max_steps = int(images_num / batch_size)

			while step <= self.max_iters_:
				batch_images, batch_labels = images[batch_range], labels[batch_range]

				sess.run(optimizer, feed_dict = {x : batch_images, y : batch_labels})

				if step % max_steps == 0:
					l, acc = sess.run ([loss, accuracy], feed_dict = {x : batch_images, y: batch_labels})

					print("[Step{}]---->{} images completed".format(step, step * batch_size))
					print("Now, the batch loss is {:.6f}, accuracy is {:.5f}".format(l, acc))

				step += 1

			print("Optimization Finished!")
			
			saver.save(sess, "./tmp/model.ckpt")
			print("Have save all variables to " + "./tmp/model.ckpt")


if __name__ == "__main__":
	# test_x = np.random.rand(8,5,5,3)
	# test_y = np.random.rand(8,4)

	test_nn = MyDNN()
	# test_nn.init_parameters(learning_rate=0.5, batch_size=2)
	# test_nn.train(test_x, test_y)
	test_nn.init_parameters(learning_rate=0.5, batch_size=2, images_num=8, max_iters=10000)
	print(test_nn.get_batch_range(100))

