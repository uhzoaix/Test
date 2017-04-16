""" handle the input issusess
This file transfer the images and input to tensorflow tensor
of size [images_number, width, height, channel]

batch_size is given by outside  parameters

width and height are required to be fixed and expected to be same for convenience

channel is 3 for rgb images, that't what we always handled

After reshaped the image, we also normalized the images tensor utilizing learning
"""

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import os
import numpy as np


def get_input(file_path):
	"""Get the input tensor for learning, 
	
	Args:
		file_path: path where saved images and its labels 
					we have several subfolders in this path, referring to differenct brandon
	"""

	# get the names of subfolders
	brandon_names = os.listdir(file_path)

	for i, name in enumerate(brandon_names):
		print(i)

	return brandon_names


def one_hot_encoding(names):
	size = len(names)

	encoding_names = [np.zeros(size) for i in range(0, size)]
	for i, vec in enumerate(encoding_names):
		vec[i] = 1

	return encoding_names


def test():
	folder_path = '..\\data\\999\\'
	# get all images path
	image_path = tf.train.match_filenames_once(folder_path + '*.jpg')
	# generate the input filename queue
	file_queue = tf.train.string_input_producer(image_path)
	# create a jpg reader and read images
	reader = tf.WholeFileReader()
	key, value = reader.read(file_queue)

	images = tf.image.decode_jpeg(value,channels=3)
	# tf.summary.image('input', images)
	merged = tf.summary.merge_all()

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		
		# visualization in Tensorboard
		file_writer = tf.summary.FileWriter('./tmp/test/', sess.graph)
		summary_str = sess.run(merged)
		file_writer.add_summary(summary_str)
		print("Summary writen")

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		images_tensor = sess.run(images)
		print(images_tensor.shape)

if __name__ == "__main__":
	test()