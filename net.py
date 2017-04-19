''' Network structure for drug brandon recogintion
This file define the network structure graph, receives preprocessed 
input and train network here
'''

import tensorflow as tf

def conv_layer(input_tensor, kernel_size, bias_size, name, activate="relu", strides=1):
	with tf.variable_scope(name) :
		w = tf.get_variable("weights", shape=kernel_size,
				initializer=tf.random_uniform_initializer())
		b = tf.get_variable("biases", shape=bias_size,
				initializer=tf.random_uniform_initializer())

		conv = tf.nn.conv2d(input_tensor, w, strides=[1, strides, strides, 1], padding="SAME")
		result = tf.nn.bias_add(conv, b)

		if activate == "relu":
			result = tf.nn.relu(result)
		else:
			print("Your activate function is not supported")
			raise ValueError

	return result


def fc_layer(input_tensor, output_size, name, activate="relu"):
	with tf.variable_scope(name):
		try :
			batch_size, feature_size = input_tensor.get_shape().as_list()
		except ValueError:
			print(input_tensor.shape)
			raise ValueError
		
		w = tf.get_variable("weights", shape=[feature_size, output_size],
				initializer=tf.random_uniform_initializer())
		b = tf.get_variable("biases", shape=[output_size],
				initializer=tf.random_uniform_initializer())

		result = tf.matmul(input_tensor, w) + b
		# now input tensor has a size of [batch_size, output_size]

		if activate == "relu":
			result = tf.nn.relu(result)
		else :
			print("Your activate function is not supported")
			raise ValueError

	return result


def pooling_layer(input_tensor, k, name):
	with tf.name_scope(name):
		pooling = tf.nn.max_pool(input_tensor, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")

	return pooling


def network(input_tensor, classes_num, is_extract_feature=False):
	"""define network structure 
		
	Args: 
		input_tensor: 
				preprocessed input tensor from input.py, of size [batch_size, image_width, image_height, channels]
		is_extract_feature: 
				bool, decide if return the feature tensor after several conv layer or the compressed softmax result

	Return:
		return the feature tensor after several conv layer or the compressed softmax result		

	"""
	batch_size, w, h, c = input_tensor.get_shape().as_list()

	# define the first layer, a convolution layer, its kernel size is defined as 'conv1_size'
	conv1_size = [3, 3, c, 16]
	bias1_size = [16]

	with tf.variable_scope("conv1") as scope:
		conv1 = conv_layer(input_tensor, conv1_size, bias1_size, name='conv1')

	# define the 2nd layer, max pooling layer 
	pooling_k = 2
	with tf.variable_scope("pooling1") as scope:
		pooling1 = pooling_layer(conv1, pooling_k, name='pool1')

	# define the 3rd layer, convolutional layer
	# 8 inputs channels and 32 outputs channels
	conv2_size = [3, 3, 16, 32]
	bias2_size = [32]
	with tf.variable_scope("conv2") as scope:
		conv2 = conv_layer(pooling1, conv2_size, bias2_size, name='conv2')

	# define 4th layer,  max pooling layer
	pooling_k2 = 2
	with tf.variable_scope("pooling2") as scope:
		pooling2 = pooling_layer(conv2, pooling_k2, name='pool2')

	# defube 5th layer, full connected layer
	# reshape the 4d tensor to tensor of size [batch size, width*height*channels] 

	pooling2 = tf.reshape(pooling2, shape=[batch_size, -1])
	with tf.variable_scope("fc1") as scope:
		fc1 = fc_layer(pooling2, output_size=128, name='fc1')

	# apply dropout, probability to keep units 
	dropout_prob = 0.75
	fc1 = tf.nn.dropout(fc1, dropout_prob)

	# define 6th layer, fc layer
	with tf.variable_scope("fc2") as scope:
		fc2 = fc_layer(fc1, output_size=classes_num, name='fc2')

	logits = fc2
	return logits


def alexnet(input_tensor, classes_num, dropout_prob=0.5):
	batch_size, w, h, c = input_tensor.get_shape().as_list()

	with tf.name_scope("alexnet"):
		# define 1st layer, convolutional layer, apply max pooling and LRN
		conv1 = conv_layer(
					name = 'conv1',
					input_tensor = input_tensor,
					kernel_size = [11, 11, c, 96],
					bias_size = [96],
					strides=4)

		conv1 = pooling_layer(
					name = 'pooling1',
					input_tensor = conv1,
					k = 2)

		conv1 = tf.nn.lrn(conv1)

		#define 2nd layer, convolutional layer, apply max pooling
		conv2 = conv_layer(
					name = 'conv2',
					input_tensor = conv1,
					kernel_size = [5, 5, 96, 256],
					bias_size = [256])

		conv2 = pooling_layer(
					name = 'pooling2',
					input_tensor = conv2,
					k = 2)

		conv2 = tf.nn.lrn(conv2)

		# define 3rd layer, convolutional layer, no max pooling
		conv3 = conv_layer(
					name = 'conv3',
					input_tensor = conv2,
					kernel_size = [3, 3, 256, 384],
					bias_size = [384])

		# define 4th layer, convolutional layer, no max pooling
		conv4 = conv_layer(
					name = 'conv4',
					input_tensor = conv3,
					kernel_size = [3, 3, 384, 384],
					bias_size = [384])

		# define the 5th layer, convolutional layer, apply max pooling
		conv5 = conv_layer(
					name = 'conv5',
					input_tensor = conv4,
					kernel_size = [3, 3, 384, 256],
					bias_size = [256])

		conv5 = pooling_layer(
					name = 'pooling3',
					input_tensor = conv5,
					k = 2)

		# define the 6th layer, full connected layer
		conv5 = tf.reshape(conv5, shape=[batch_size, -1])

		fc1 = fc_layer(
					name = 'fc1',
					input_tensor = conv5,
					output_size = 4096
					)

		fc1 = tf.nn.dropout(fc1, dropout_prob)

		# define the 7th layer, full connected layer
		fc2 = fc_layer(
					name = 'fc2',
					input_tensor = fc1,
					output_size = 4096)

		fc2 = tf.nn.dropout(fc2, dropout_prob)

		# calculate logits, 
		logits = fc_layer(
					name = 'logits',
					input_tensor = fc2,
					output_size = classes_num)

	return logits

def loss(_logits, _labels, regularization_lambda):
	"""use tf.nn.softmax_cross_entropy_with_logits to c
	Args:
		logits: unscaled probability distribution for each image in batch ,
				of size [batch_size, label_size ]

		labels: 1d tensor, usually in one-hot format

	"""
	regularization_loss = 0
	if regularization_lambda != 0:
		var_list = tf.global_variables()

		for var in var_list:
			regularization_loss += tf.nn.l2_loss(var)

	regularization_loss = regularization_lambda * regularization_loss
	# calculate a loss tensor of size [batch_size]
	batch_loss = tf.nn.softmax_cross_entropy_with_logits(logits=_logits, labels=_labels)
	# calculate the mean of the loss tensor as the final output of loss
	return tf.reduce_mean(batch_loss) + regularization_loss


if __name__ == "__main__":
	# for test
	print("Only for test, you won't see this message unless you directly run this file")
	with tf.Graph().as_default():
		x = tf.placeholder(tf.float32, [20,200,200,3])
		y = tf.placeholder(tf.float32, [20,5])

		test_logits = alexnet(x, 5)

		loss = loss(test_logits, y, 0.05)
