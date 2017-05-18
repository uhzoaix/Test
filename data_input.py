""" handle the input issusess
This file transfer the images and input to tensorflow tensor
of size [images_number, width, height, channel]

batch_size is given by outside  parameters

width and height are required to be fixed and expected to be same for convenience

channel is 3 for rgb images, that't what we always handled

After reshaped the image, we also normalized the images tensor utilizing learning
"""

import tensorflow as tf
import os
import numpy as np

def one_hot_encoding(names):
	size = len(names)

	encoding_names = [np.zeros(size) for i in range(0, size)]
	for i, vec in enumerate(encoding_names):
		vec[i] = 1

	return encoding_names


def test():
	path = "../data/cat/"

	images_filename = [None] * 3600
	labels_filename = [None] * 3600

	with open(path+"file_list.txt", 'r') as f:
		for i, line in enumerate(f.readlines()):
			line = line.replace('\n', '')
			images_filename[i] = path + line + ".jpg"
			labels_filename[i] = int(line[0])

		print(images_filename[:20])
		print(labels_filename[:20])

	print()

	images_filename = tf.convert_to_tensor(images_filename, dtype=tf.string)
	labels_filename = tf.convert_to_tensor(labels_filename, dtype=tf.int32)

	input_queue = tf.train.slice_input_producer(
		[images_filename, labels_filename],
		shuffle=True)

	file_content = tf.read_file(input_queue[0])
	train_image = tf.image.decode_jpeg(file_content, channels=3)
	train_label = input_queue[1]

	train_image = tf.image.convert_image_dtype(train_image, tf.float32)
	train_image = tf.image.resize_image_with_crop_or_pad(train_image, 200, 200)
	train_image.set_shape([200,200,3])

	train_image_batch, train_label_batch = tf.train.batch(
		[train_image, train_label],
		batch_size=20)

	with tf.Session() as sess:
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		for i in range(10):
			print("Image batch shape: ", sess.run(train_image_batch).shape)
			print(sess.run(train_label_batch))

		coord.request_stop()
		coord.join(threads)


def get_images_batch_with_labels(data_path, shape, channels, batch_size, data_size):
	images_filename = [None] * data_size
	labels_filename = [None] * data_size

	# get images filename from file_list.txt,
	with open(data_path+"file_list.txt", 'r') as f:
		for i, line in enumerate(f.readlines()):
			if i >= data_size:
				# don't use the rest data
				break

			line = line.replace('\n', '')
			images_filename[i] = data_path + line + ".jpg"
			labels_filename[i] = int(line[0])

		# remove all Nones from lists 
		if None in images_filename and None in labels_filename:
			images_filename = [val for val in images_filename if not val == None]
			labels_filename = [val for val in labels_filename if not val == None]

	# create filenames input queue, and return image and label batch
	images_filename = tf.convert_to_tensor(images_filename, dtype=tf.string)
	labels_filename = tf.convert_to_tensor(labels_filename, dtype=tf.int32)

	input_queue = tf.train.slice_input_producer(
		[images_filename, labels_filename],
		shuffle=True)

	file_content = tf.read_file(input_queue[0])
	image = tf.image.decode_jpeg(file_content, channels=3)
	label = input_queue[1]

	image = image_preprocessing(image, shape=shape, channels=channels, distorted=True)
	image_batch, label_batch = tf.train.batch(
		[image, label],
		batch_size=batch_size)

	return image_batch, label_batch


def get_input_images_batch(image_path, shape, channels=3, batch_size=20, num_threads=1, min_after_dequeue=1000):
	# find all jpg files in the given path
	w,h = shape

	paths = tf.train.match_filenames_once(image_path + "*.jpg")
	file_queue = tf.train.string_input_producer(paths)

	# create files reader
	reader = tf.WholeFileReader()
	key, value = reader.read(file_queue)

	images = tf.image.decode_jpeg(value, channels=3)
	images = tf.image.resize_images(images, shape)
	images.set_shape((w,h,channels))

	image_batch = tf.train.shuffle_batch(
		[images],
		batch_size=batch_size,
		num_threads=num_threads,
		capacity=min_after_dequeue + 3 * batch_size,
		min_after_dequeue=min_after_dequeue)

	return image_batch


def image_preprocessing(decode_image, shape, channels, distorted=False):
	w,h = shape
	# convert the image to float
	decode_image = tf.image.convert_image_dtype(decode_image, tf.float32)
	# resize the image, and convert its data type to float
	decode_image = tf.image.resize_image_with_crop_or_pad(decode_image, w, h)
	decode_image.set_shape([w, h, channels])

	if distorted:
		decode_image = tf.image.random_flip_left_right(decode_image)
		decode_image = tf.image.random_brightness(decode_image, max_delta=63)
		decode_image = tf.image.random_contrast(decode_image, lower=0.2, upper=1.8)
	# standarization
	decode_image = tf.image.per_image_standardization(decode_image)

	return decode_image


def concatenate_data():
	import shutil
	path = "../data/"
	if not os.path.exists(path + "cat/"):
		os.makedirs(path + "cat/")

	target_path = path + "cat/"
	brand_names = os.listdir(path)
	brand_names.remove("cat")
	image_num = 720
	for label, brand in enumerate(brand_names):
		temp_path = path + brand + "/"
		# copy the image files to 'cat' folder
		for i in range(1,image_num+1):
			image_name = str(i) + ".jpg"
			shutil.copy2(temp_path+image_name, target_path + "{}_{}.jpg".format(label, i))
			print("Copy file: " + temp_path+image_name + " to " +target_path + "{}_{}.jpg".format(label, i))
			# create the label file
			with open(target_path + "{}_{}.txt".format(label, i), 'w') as f:
				print(label, file=f)
				print("Write file: " + target_path + "{}_{}.txt".format(label, i))


if __name__ == "__main__":
	# just for test
	train_size=2000
	train_image_batch, train_label_batch, test_image_batch, test_label_batch = get_images_batch_with_labels(
		data_path= "../data/cat/",
		shape=[200, 200],
		channels=3,
		batch_size=20,
		train_size=train_size,
		test_size=0)

	print("Got Batch")
	with tf.Session() as sess:
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		batch = sess.run(train_image_batch)
		label = sess.run(train_label_batch)
		print("Train Batch shape: ", batch.shape)
		print("Train Lable: ", label)

		coord.request_stop()
		coord.join(threads)