import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import os


dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/cifar_10_tf_train_test.pkl"
data_file = file(filename, "rb")
train_x, train_y, test_x, test_y = pickle.load(data_file)
data_file.close()

float_train_x = train_x.astype(np.float32)/255
float_test_x = test_x.astype(np.float32)/255
mean_train_x = np.mean(float_train_x, axis = 0)
mean_test_x = np.mean(float_test_x, axis = 0)

filename2 = dir_path + "/result7.txt"
data_file2 = file(filename2, "rb")
parameter = pickle.load(data_file2)
data_file.close()

for i in range(len(float_train_x)):
	float_train_x[i] -= mean_train_x

for i in range(len(float_test_x)):
	float_test_x[i] -= mean_test_x


# shuffle function
def shufflelists(lists):
	rv = np.random.permutation(len(lists[1]))
	output = []
	for l in lists:
		output.append(l[rv])
	return output


#use placeholder for nodes
x = tf.placeholder(shape = [None, 32, 32, 3], dtype = tf.float32)
y = tf.placeholder(shape = [None], dtype = tf.int64)

# first convolution layer
conv_filter_w1 = tf.Variable(parameter[0])
conv_filter_b1 = tf.Variable(parameter[1])

# activation
feature_maps1 = (tf.nn.conv2d(x, conv_filter_w1, strides = [1,1,1,1], padding = 'VALID') + conv_filter_b1)

relu_feature_maps1 = tf.nn.relu(feature_maps1)

# pooling
max_pool1 = tf.nn.max_pool(relu_feature_maps1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

# second convolution layer
conv_filter_w2 = tf.Variable(parameter[2])
conv_filter_b2 = tf.Variable(parameter[3])

# activation
feature_maps2 = (tf.nn.conv2d(max_pool1, conv_filter_w2, strides = [1,1,1,1], padding = 'VALID') + conv_filter_b2)

relu_feature_maps2 = tf.nn.relu(feature_maps2)


# pooling
max_pool2 = tf.nn.max_pool(relu_feature_maps2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

# Third convolution layer
conv_filter_w3 = tf.Variable(parameter[4])
conv_filter_b3 = tf.Variable(parameter[5])

relu_feature_maps3 = tf.nn.conv2d(max_pool2, conv_filter_w3, strides = [1,1,1,1], padding = 'VALID') + conv_filter_b3


# batch normalization
batch_mean, batch_var = tf.nn.moments(relu_feature_maps3, [0, 1, 2], keep_dims = True)
shift = tf.Variable(tf.zeros([64]))
scale = tf.Variable(tf.ones([64]))
epsilon = 1e-3
BN_out = tf.nn.batch_normalization(relu_feature_maps3, batch_mean, batch_var, shift, scale, epsilon)

relu_BN_maps3 = tf.nn.relu(BN_out)

# fully connected layer
flat = tf.reshape(relu_BN_maps3, [-1, 3*3*64])
fc_w1 = tf.Variable(parameter[6])
fc_b1 = tf.Variable(parameter[7])

logit_value = tf.matmul(flat, fc_w1) + fc_b1

pred = tf.nn.softmax(logit_value)

# loss function
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logit_value)

zhuergou = tf.Variable(tf.constant(0.001), dtype = tf.float32)
train_step = tf.train.AdamOptimizer(learning_rate = zhuergou).minimize(loss)

update_rate = tf.assign(zhuergou, 0.85*zhuergou)

predict_op = tf.argmax(logit_value, 1)
bool_pred = tf.equal(y, predict_op)

# accuracy
accuracy = tf.reduce_mean(tf.cast(bool_pred, tf.float32))

tf.get_collection('validation_nodes')
tf.add_to_collection('validation_nodes', x )
tf.add_to_collection('validation_nodes', predict_op)


saver = tf.train.Saver()

model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)
	
	for epoch in range(4):
		b_idx = 0
		b_size = 50
		count = 1
		'''
		float_train_x, train_y = shufflelists([float_train_x, train_y])
		'''
		while (b_idx + b_size) <= float_train_x.shape[0]:
			session.run(train_step, feed_dict = {x: float_train_x[b_idx: b_idx + b_size], y : train_y[b_idx: b_idx + b_size]})
			b_idx += b_size
			count += 1
			if count % 10 == 0:
				session.run(update_rate)
			if count % 50 == 0:
				rst = []
				rst.append(conv_filter_w1.eval())
				rst.append(conv_filter_b1.eval())
				rst.append(conv_filter_w2.eval())
				rst.append(conv_filter_b2.eval())
				rst.append(conv_filter_w3.eval())
				rst.append(conv_filter_b3.eval())
				rst.append(fc_w1.eval())
				rst.append(fc_b1.eval())
				filehandler = open("result8.txt", "wb")
				pickle.dump(rst, filehandler, protocol = 2)
				filehandler.close()
				print "###################update#################"
			'''
			res1 = accuracy.eval({x: float_train_x, y: train_y})
			print res1
			'''
			res2 = accuracy.eval({x: float_test_x, y: test_y})
			print res2
			print "###"
		
	print ("Optimize finished")
	'''
	res1 = accuracy.eval({x: float_train_x, y: train_y})
	print res1
	'''
	res2 = accuracy.eval({x: float_test_x, y: test_y})
	print res2
	print "###"
	save_path = saver.save(session, "./my_model")

