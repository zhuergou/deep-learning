import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/transform.pkl"
data_file = file(filename, "rb")
raw_x, raw_y = pickle.load(data_file)
data_file.close()

raw_x = raw_x.astype(np.float32)
raw_x -= np.mean(raw_x, axis = (2, 3, 4),  keepdims = True)
raw_x /= np.std(raw_x, axis = (2, 3, 4), keepdims = True)

train_x = raw_x[0:7000]
test_x = raw_x[7000:8000]
train_y = raw_y[0:7000]
test_y = raw_y[7000:8000]

x = tf.placeholder(shape = [None, 10, 64, 64, 3], dtype = tf.float32)
y = tf.placeholder(shape = [None, 10, 7, 2], dtype = tf.float32)

conv_filter_w1 = tf.Variable(tf.truncated_normal(shape = [5, 5, 3, 8], stddev = 0.02))

conv_filter_b1 = tf.Variable(tf.truncated_normal(shape = [8], stddev = 0.02))

conv_filter_w2 = tf.Variable(tf.truncated_normal(shape = [5, 5, 8, 8], stddev = 0.02))

conv_filter_b2 = tf.Variable(tf.truncated_normal(shape = [8], stddev = 0.02))
conv_filter_w3 = tf.Variable(tf.truncated_normal(shape = [3, 3, 8, 16], stddev = 0.02))

conv_filter_b3 = tf.Variable(tf.truncated_normal(shape = [16], stddev = 0.02))

def get_rnn_input(x):

	relu_feature_maps1 = tf.nn.relu(tf.nn.conv2d(x, conv_filter_w1, strides = [1, 1, 1, 1], padding = 'VALID') + conv_filter_b1)

	max_pool1 = tf.nn.max_pool(relu_feature_maps1, ksize = [1, 3, 3, 1], strides = [1, 3, 3, 1], padding = 'VALID')


	relu_feature_maps2 = tf.nn.relu(tf.nn.conv2d(max_pool1, conv_filter_w2, strides = [1, 1, 1, 1], padding = 'VALID') + conv_filter_b2)

	max_pool2 = tf.nn.max_pool(relu_feature_maps2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')


	relu_feature_maps3 = tf.nn.relu(tf.nn.conv2d(max_pool2, conv_filter_w3, strides = [1,1,1,1], padding = 'VALID') + conv_filter_b3)

	max_pool3 = tf.nn.max_pool(relu_feature_maps3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

	flat = tf.reshape(max_pool3, [-1, 3*3*16])
	return flat


rnn_input = tf.expand_dims(get_rnn_input(x[:,0,:,:,:]), 1)

for i in range(1, 10):
	rnn_input = tf.concat([rnn_input, tf.expand_dims(get_rnn_input(x[:,i,:,:,:]), 1)], 1)


lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units = 128)
'''
_init_state = lstm_cell.zero_state(None, dtype = tf.float32)
'''
outputs, state = tf.nn.dynamic_rnn(lstm_cell, rnn_input,  dtype = tf.float32)

w_fc = tf.Variable(tf.truncated_normal(shape = [128, 14], stddev = 0.02))
b_fc = tf.Variable(tf.truncated_normal(shape = [14], stddev = 0.02))


for i in range(10):
	temp = tf.expand_dims(tf.matmul(outputs[:,i,:], w_fc) + b_fc, 1)
	if i == 0:
		final = temp
	else:
		final = tf.concat([final, temp], 1)

print (tf.shape(final))

predict_op = tf.reshape(final,[-1, 10, 7, 2])

cost = tf.sqrt(2 * tf.reduce_mean((predict_op - y)*(predict_op - y)))
train_op = tf.train.AdamOptimizer(0.01).minimize(cost)

if not os.path.exists('tmp/'):
	os.mkdir('tmp/')

tf.get_collection('validation_nodes')
tf.add_to_collection('validation_nodes', x)
tf.add_to_collection('validation_nodes', y)
tf.add_to_collection('validation_nodes', predict_op)

saver =tf.train.Saver()

with tf.Session() as session:

	if os.path.exists('tmp/checkpoint'):
		saver.restore(session, './my_model')
	else:
		model = tf.global_variables_initializer()
		session.run(model)
	'''
	for epoch in range(5):
		b_idx = 0
		b_size = 50
		count = 0
	
		while (b_idx + b_size) < 7000:
			session.run(train_op, feed_dict = {x: train_x[b_idx: b_idx + b_size], y: train_y[b_idx: b_idx + b_size]})
			b_idx += b_size
			count += 1
			
			if count % 100 == 0:
				rst = []
				rst.append(conv_filter_w1.eval())
				rst.append(conv_filter_b1.eval())
				rst.append(conv_filter_w2.eval())
				rst.append(conv_filter_b2.eval())
				rst.append(conv_filter_w3.eval())
				rst.append(conv_filter_b3.eval())
				rst.append(_init_state.eval())
				rst.append(w_fc.eval())
				rst.append(b_fc.eval())
				filehandler = open("result.txt", "wb")
				pickledump(rst, filehandler, protocol = 2)
				filehandler.close()
				print "###################"
				
				save_path = saver.save(session, 'tmp/model.ckpt')
				print "########################"
				
				res1 = cost.eval({x: train_x, y: train_y})
				print res1

				res2 = cost.eval({x: test_x, y: test_y})
				print res2
				
			
			print count
			print "#########"
	
		save_path = saver.save(session, './my_model')
		print "########################"
	'''
	res1 = cost.eval({x: train_x, y: train_y})
	print res1

	res2 = cost.eval({x: test_x, y: test_y})
	print res2
		

	print "########################"
