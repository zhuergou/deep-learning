import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
import pickle

def shufflelists(lists):
	rv = np.random.permutation(len(lists[1]))
	output = []
	for l in lists:
		output.append(l[rv])
	return output

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/new.pkl"
data_file = file(filename,"rb")
train_x, train_y, test_x, test_y = pickle.load(data_file)
data_file.close()

filename = dir_path + "/extra_parameter.txt"
data_file = file(filename, "rb")
parameter = pickle.load(data_file)
data_file.close()

train_x_new = np.reshape(train_x, [4000, 28, 28])
test_x_new = np.reshape(test_x, [1000, 28, 28]) 

train_x_new = (train_x_new - np.mean(train_x_new))/np.std(train_x_new)
test_x_new = (test_x_new - np.mean(test_x_new))/np.std(test_x_new)

print train_x_new.shape
print test_x_new.shape
x = tf.placeholder("float", [None, 28, 28])
y = tf.placeholder("float", [None, 1])
'''
W1 = tf.Variable(tf.truncated_normal(shape = [5,5], stddev = 0.03))
W1_0 = tf.Variable(tf.truncated_normal([24,24], stddev = 0.03))
W2 = tf.Variable(tf.truncated_normal(shape = [64, 1], stddev = 0.03))
W2_0 = tf.Variable(tf.truncated_normal(shape = [1, 1], stddev = 0.03))
'''
W1 = tf.Variable(parameter[0])
W1_0 = tf.Variable(parameter[1])
W2 = tf.Variable(parameter[2])
W2_0 = tf.Variable(parameter[3])

learning_rate = tf.Variable(0.003)

def get_conv(x, W1, W1_0):
	for j in range(24):
		for i in range(24):
			if i == 0:
				rst = tf.expand_dims(tf.reduce_sum(tf.slice(x, [0, i, j], [-1, 5, 5])*W1, [1, 2]), 1)
			else:
				rst = tf.concat([rst, tf.expand_dims(tf.reduce_sum(tf.slice(x, [0, i, j], [-1, 5, 5])*W1, [1, 2]), 1)], 1)
		if j == 0:
			black = tf.expand_dims(rst, 2)
		else:
			black = tf.concat([black, tf.expand_dims(rst, 2)], 2)
	
	return black

conv_layer = get_conv(x, W1, W1_0)
activate_layer = tf.nn.relu(conv_layer)

def get_pool(activate_layer):
	for j in range(8):
		for i in range(8):
			if i == 0:
				rst = tf.expand_dims(tf.reduce_max(tf.slice(activate_layer, [0, 3*i, 3*j], [-1, 3, 3]), [1, 2]), 1)
			else:
				rst = tf.concat([rst, tf.expand_dims(tf.reduce_max(tf.slice(activate_layer, [0, 3*i, 3*j], [-1, 3, 3]), [1, 2]), 1)],1)
		if j == 0:
			black = tf.expand_dims(rst, 2)
		else:
			black = tf.concat([black, tf.expand_dims(rst, 2)], 2)
	return black

pool_rst = get_pool(activate_layer)
print pool_rst.get_shape().as_list()

pool_rst_flat = tf.reshape(pool_rst, [-1, 64])

pred = tf.sigmoid(tf.matmul(pool_rst_flat, W2) + W2_0)

## Cost function
cost = tf.reduce_mean(0.5* tf.reduce_sum((y - pred)**2, reduction_indices = 1))

W2_grad = pred**2 * pool_rst_flat *(pred - y)
W2_0_grad = pred **2 *(pred - y)

## here may have issue
pool_grad = pred**2 * tf.transpose(tf.tile(W2, [1, tf.shape(pred)[0]])) *(pred - y)

print pool_grad.get_shape().as_list()

def get_pool_grad(pool_grad):
	for i in range(8):
		if i == 0:
			rst = tf.expand_dims(tf.slice(pool_grad, [0, 0], [-1, 8]), 1)
		else:
			rst = tf.concat([rst, tf.expand_dims(tf.slice(pool_grad, [0, 8*i], [-1, 8]), 1)], 1)

	return rst


pool_before_flat_grad = get_pool_grad(pool_grad)
print pool_before_flat_grad.get_shape().as_list()

def get_A_grad(pool_before_flat_grad, activate_layer):
	'''
	A_grad = tf.Variable(tf.zeros([50, 24, 24]))
	for k in range(50):
		for i in range(8):
			for j in range(8):
				max_value = -1000
				index_a = -1
				index_b = -1
				for m in range(2):
					for n in range(2):
						if activate_layer[k, 3*i+m, 3*j+n] > max_value:
							max_value = activate_layer[k, 3*i+m, 3*j + n]
							index_a = 3*i + m
							index_b = 3*j + n
				A_grad[k, index_a, index_b] = pool_before_flat_grad[k, i, j]
	'''
	for i in range(8):
		for j in range(8):
			if  (j==0):
				rst = (tf.cast(tf.equal(activate_layer[:, (3*i): (3*i+3), (3*j):(3*j+3)], tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(tf.reduce_max(activate_layer[:, (3*i):(3*i+3), (3*j):(3*j+3)],[1,2]),1),[1,3]),2),[1,1,3])), tf.float32)* tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(pool_before_flat_grad[:,i,j],1),[1,3]),2),[1,1,3]))
				print rst.get_shape().as_list()
			else:
				rst = tf.concat([rst, tf.cast(tf.equal(activate_layer[:, (3*i): (3*i+3), (3*j):(3*j+3)], tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(tf.reduce_max(activate_layer[:, (3*i):(3*i+3), (3*j):(3*j+3)],[1,2]), 1),[1,3]),2),[1,1,3])), tf.float32)* tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(pool_before_flat_grad[:,i,j],1),[1,3]), 2),[1,1,3])], 2)
		if i == 0:
			black = rst
		else:
			black = tf.concat([black, rst], 1)
	return black


A_grad = get_A_grad(pool_before_flat_grad, activate_layer)
print A_grad.get_shape().as_list()
				
def get_C_grad(A_grad, conv_layer):
	'''
	C_grad = tf.Variable(tf.zeros([50, 24, 24]))
	for i in range(24):
		for j in range(24):
			for k in range(50):
				if conv_layer[k, i, j] > 0:
					C_grad[k, i, j] = A_grad[k, i, j]
	'''
	rst = tf.cast(tf.greater(conv_layer, 0), tf.float32)*A_grad
	return rst

C_grad = get_C_grad(A_grad, conv_layer)
print C_grad.get_shape().as_list()

def get_W_grad(C_grad, x):
	'''
	W_grad = tf.Variable(tf.zeros([50, 5, 5]))
	for i in range(5):
		for j in range(5):
			for k in range(50):
				tmp = 0
				for m in range(24):
					for n in range(24):
						tmp = tmp + x[k, i+m, j+n] *C_grad[m][n]
				W_grad[k, i, j] += tmp
	'''
	for j in range(5):
		for i in range(5):
			if i == 0:
				rst = tf.expand_dims(tf.reduce_sum(x[:, i:(i+24), j:(j+24)]* C_grad[:,0:24, 0:24],[1,2]),1)
			else:
				rst = tf.concat([rst, tf.expand_dims(tf.reduce_sum(x[:, i:(i+24), j:(j+24)]* C_grad[:,0:24, 0:24],[1,2]), 1)], 1)
		if j == 0:
			black = tf.expand_dims(rst, 2)
		else:
			black = tf.concat([black, tf.expand_dims(rst,2)], 2)

	return black

W1_grad = get_W_grad(C_grad, x)
print W1_grad.get_shape().as_list()

W1_0_grad = C_grad
print W1_0_grad.get_shape().as_list()

update_step = [tf.assign(W1, W1 - learning_rate* tf.reduce_mean(W1_grad, 0)), tf.assign(W1_0, W1_0 - learning_rate* tf.reduce_mean(W1_0_grad, 0)), tf.assign(W2, W2 - learning_rate* tf.expand_dims(tf.reduce_mean(W2_grad, 0),1)), tf.assign(W2_0, W2_0 - learning_rate* tf.expand_dims(tf.reduce_mean(W2_0_grad, 0), 1))]


correct_prediction = tf.greater((pred - 0.5)*(y-0.5), 0)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

model = tf.global_variables_initializer()
with tf.Session() as session:
	session.run(model)
	for epoch in range(30):
		b_idx = 0
		b_size = 50
		count = 1
		train_x_new, train_y = shufflelists([train_x_new, train_y])
		while (b_idx + b_size) <= train_x_new.shape[0]:
			'''
			print train_x_new[b_idx:b_idx+b_size].shape
			print train_y[b_idx: b_idx+b_size].shape
			'''
			session.run(update_step, feed_dict = {x: train_x_new[b_idx: b_idx + b_size], y: train_y[b_idx: b_idx + b_size]})
			b_idx += b_size
			count = count + 1
			
			rst1 = accuracy.eval({x: train_x_new, y: train_y})
			print rst1
			
			rst2 = accuracy.eval({x: test_x_new, y: test_y})
			print rst2
			
			print "#####"
			if count % 50 == 0:
				thetha = []
				thetha.append(W1.eval())
				thetha.append(W1_0.eval())
				thetha.append(W2.eval())
				thetha.append(W2_0.eval())
				filehandler = open("extra_parameteri1.txt", "wb")
				pickle.dump(thetha, filehandler)
				filehandler.close()
				print "###########update#########"
