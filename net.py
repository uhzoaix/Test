''' Network structure for drug brandon recogintion
This file define the network structure graph, receives preprocessed 
input and train network here
'''

import tensorflow as tf

def conv_layer(input_tensor, kernel_size, bias_size, activate="relu", strides=1):
	w = tf.get_variable("weights", shape=kernel_size)
	b = tf.get_variable("biases", shape=bias_size)

	conv = tf.nn.conv2d(input_tensor, w, strides=[1, strides, strides, 1], padding="SAME")
	result = tf.nn.bias_add(conv, b)

	if activate == "relu":
		result = tf.nn.relu(result)
	else:
		print("Your activate function not suppory")
		raise ValueError

	return result


def fc_layer(input_tensor, output_size, activate="relu"):
	batch_size, feature_size = input_tensor.get_shape().as_list()
	
	w = tf.get_variable("weights", shape=[feature_size, output_size])
	b = tf.get_variable("biases", shape=[output_size])

	result = tf.matmul(input_tensor, w) + b
	# now input tensor has a size of [batch_size, output_size]

	if activate == "relu":
		result = tf.nn.relu(result)
	else :
		print("Your activate function not suppory")
		raise ValueError

	return result


def pooling_layer(input_tensor, k):
	pooling = tf.nn.max_pool(input_tensor, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")

	return pooling


def network(input_tensor, is_extract_feature=False):
	"""define network structure 
		
	Args: 
		input_tensor: 
				preprocessed input tensor from input.py, of size [batch_size, image_width, image_height, channels]
		is_extract_feature: 
				bool, decide if return the feature tensor after several conv layer or the compressed softmax result

	Return:
		return the feature tensor after several conv layer or the compressed softmax result		

	"""
	size_list = [size.value for size in input_tensor.get_shape()]
	batch_size, w, h, c = size_list

	# define the first layer, a convolution layer, its kernel size is defined as 'conv1_size'
	conv1_size = [3, 3, c, 16]
	bias1_size = [16]
	with tf.variable_scope("conv1") as scope:
		conv1 = conv_layer(input_tensor, conv1_size, bias1_size)

	# define the 2nd layer, max pooling layer 
	pooling_k = 2
	with tf.variable_scope("pooling1") as scope:
		pooling1 = pooling_layer(conv1, pooling_k)

	# define the 3rd layer, convolutional layer
	# 8 inputs channels and 32 outputs channels
	conv2_size = [3, 3, 16, 32]
	bias2_size = [32]
	with tf.variable_scope("conv2") as scope:
		conv2 = conv_layer(pooling1, conv2_size, bias2_size)

	# define 4th layer,  max pooling layer
	pooling_k2 = 2
	with tf.variable_scope("pooling2") as scope:
		pooling2 = pooling_layer(conv2, pooling_k2)

	# defube 5th layer, full connected layer
	# reshape the 4d tensor to tensor of size [batch size, width*height*channels] 

	pooling2 = tf.reshape(pooling2, shape=[batch_size, -1])
	with tf.variable_scope("fc1") as scope:
		fc1 = fc_layer(pooling2, output_size=128)

	# apply dropout, probability to keep units 
	dropout_prob = 0.75
	fc1 = tf.nn.dropout(fc1, dropout_prob)

	# define 6th layer, fc layer
	with tf.variable_scope("fc2") as scope:
		fc2 = fc_layer(fc1, output_size=5)

	logits = fc2
	return logits


def loss(_logits, _labels):
	"""use tf.nn.softmax_cross_entropy_with_logits to c
	Args:
		logits: unscaled probability distribution for each image in batch ,
				of size [batch_size, label_size ]

		labels: 1d tensor, usually in one-hot format

	"""
		
	# calculate a loss tensor of size [batch_size]
	batch_loss = tf.nn.softmax_cross_entropy_with_logits(logits=_logits, labels=_labels)
	# calculate the mean of the loss tensor as the final output of loss
	return tf.reduce_mean(batch_loss)


if __name__ == "__main__":
	# for test
	print("Only for test, you won't see this message unless you directly run this file")
