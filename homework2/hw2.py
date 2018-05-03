import matplotlib.image as mpimg
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

# shuffle function
def shufflelists(lists):
	rv = np.random.permutation(len(lists[1]))
	output = []
	for l in lists:
		output.append(l[rv])
	return output

# Parameters
learning_rate = 0.01
training_epochs = 4

# true x_value
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/train_data/00001.jpg"
x_value = mpimg.imread(filename).reshape(1, 784)
for i in range(2, 50001):
	if (i >= 1) and (i <= 9):
		filename = dir_path + "/train_data/0000" + str(i) + ".jpg"
	elif (i >= 10) and (i <= 99):
		filename = dir_path + "/train_data/000" + str(i) + ".jpg"
	elif (i >= 100) and (i <= 999):
		filename = dir_path + "/train_data/00" + str(i) + ".jpg"
	elif (i >= 1000) and (i <= 9999):
		filename = dir_path + "/train_data/0" + str(i) + ".jpg"
	else:
		filename = dir_path + "/train_data/" + str(i) + ".jpg"
	x_value = np.vstack((x_value, mpimg.imread(filename).reshape(1, 784)))


# true x_test
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/test_data/00001.jpg"
x_test = mpimg.imread(filename).reshape(1, 784)
for i in range(2, 5001):
	if (i >= 1) and (i <= 9):
		filename = dir_path + "/test_data/0000" + str(i) + ".jpg"
	elif (i >= 10) and (i <= 99):
		filename = dir_path + "/test_data/000" + str(i) + ".jpg"
	elif (i >= 100) and (i <= 999):
		filename = dir_path + "/test_data/00" + str(i) + ".jpg"
	else:
		filename = dir_path + "/test_data/0" + str(i) + ".jpg"
	x_test = np.vstack((x_test, mpimg.imread(filename).reshape(1, 784)))


# true y_value
filename2 = dir_path + "/labels/train_label.txt"
y_pre = np.loadtxt(filename2)
y_value = np.zeros((50000, 10))
for i in range(50000):
	y_value[i][int(y_pre[i])] = 1


# true y_test
filename2 = dir_path + "/labels/test_label.txt"
y_pre = np.loadtxt(filename2)
y_test = np.zeros((5000, 10))
for i in range(5000):
	y_test[i][int(y_pre[i])] = 1

print("Loading Image Finished\n")

# use placeholder for nodes
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

# set model weights
W1 = tf.Variable(initial_value = tf.truncated_normal(shape = [784, 100], stddev = 0.02))
W1_0 = tf.Variable(initial_value = tf.zeros([100, 1]))
W2 = tf.Variable(initial_value = tf.truncated_normal(shape = [100, 100], stddev = 0.02))
W2_0 = tf.Variable(initial_value = tf.zeros([100, 1]))
W3 = tf.Variable(initial_value = tf.truncated_normal(shape = [100, 10], stddev = 0.02))
W3_0 = tf.Variable(initial_value = tf.zeros([10, 1]))


# Construct model
Z1 = tf.matmul(x, W1) + tf.transpose(W1_0)
H1 = tf.nn.relu(tf.matmul(x, W1) + tf.transpose(W1_0))
Z2 = tf.matmul(H1, W2) + tf.transpose(W2_0)
H2 = tf.nn.relu(tf.matmul(H1, W2) + tf.transpose(W2_0))
pred = tf.nn.softmax(tf.matmul(H2, W3) + tf.transpose(W3_0))


# Cost function
cost = tf.reduce_mean(0.5 * tf.reduce_sum((y - pred)**2, reduction_indices = 1))

# Compute gradient for W3
def get_W3_grad(pred, H2, y):
	for i in range(10):
		tmp = tf.reduce_sum(- tf.transpose(tf.reshape(tf.tile(pred[:,i],[10]), [10, tf.shape(pred)[0]])) * pred*(pred - y),  1) + pred[:,i]*(1-pred[:,i])*(pred[:,i]-y[:,i]) + pred[:,i]*pred[:,i]*(pred[:,i]-y[:,i])
		if i == 0:
			rst = tf.expand_dims(tf.transpose(tf.reshape(tf.tile(tmp, [100]), [100, tf.shape(tmp)[0]])) * H2, 2)
		else:
			rst = tf.concat([rst, tf.expand_dims(tf.transpose(tf.reshape(tf.tile(tmp, [100]), [100, tf.shape(tmp)[0]])) * H2, 2)], 2)

	return rst

W3_grad = get_W3_grad(pred, H2, y)
	
# Compute gradient for W3_0 
def get_W3_0_grad(pred, y):
	for i in range(10):
		tmp = tf.reduce_sum(- tf.transpose(tf.reshape(tf.tile(pred[:,i],[10]), [10, tf.shape(pred)[0]])) * pred*(pred - y),  1) + pred[:,i]*(1-pred[:,i])*(pred[:,i]-y[:,i]) + pred[:,i]*pred[:,i]*(pred[:,i]-y[:,i])
		if i == 0:
			rst = tf.expand_dims(tmp, 1)
		else:
			rst = tf.concat([rst, tf.expand_dims(tmp, 1)], 1)
	
	return rst

W3_0_grad = get_W3_0_grad(pred, y)

# Compute gradient for H2
H2_grad = tf.matmul(pred *(pred - y), tf.transpose(W3)) - tf.transpose(tf.reshape(tf.tile(tf.reduce_sum(pred*(pred - y), axis = 1), [100]), [100, tf.shape(pred)[0]])) * tf.matmul(pred, tf.transpose(W3))

# Compute gradient for W2
def get_W2_grad(H1, H2_grad, Z2):
	rst = tf.expand_dims(tf.transpose(tf.reshape(tf.tile((Z2[:,0] - tf.minimum(0.0, Z2[:,0]))/Z2[:,0] * H2_grad[:,0], [100]), [100, tf.shape(pred)[0]])) * H1, 2)
	for i in range(1, 100):
		rst = tf.concat([rst, tf.expand_dims(tf.transpose(tf.reshape(tf.tile((Z2[:,i] - tf.minimum(0.0, Z2[:,i]))/Z2[:,i] *H2_grad[:,i], [100]), [100, tf.shape(pred)[0]])) * H1, 2)], 2)
	return rst
	
W2_grad = get_W2_grad(H1, H2_grad, Z2)

# Compute gradient for W2_0
def get_W2_0_grad(H2_grad, Z2):
	rst = tf.expand_dims((Z2[:,0] - tf.minimum(0.0, Z2[:,0]))/Z2[:,0] * H2_grad[:,0], 1)
	for i in range(1, 100):
		rst = tf.concat([rst, tf.expand_dims((Z2[:,i] - tf.minimum(0.0, Z2[:,i]))/Z2[:,i] *H2_grad[:,i] , 1)], 1)
	return rst

W2_0_grad = get_W2_0_grad(H2_grad, Z2)


# Compute gradient for H1
def get_H1_grad(W2, H2_grad, Z2):
	rst = tf.expand_dims((Z2[:,0] - tf.minimum(0.0, Z2[:,0]))/Z2[:,0] * H2_grad[:,0], 1)
	for i in range(1, 100):
		rst = tf.concat([rst, tf.expand_dims((Z2[:,i] - tf.minimum(0.0, Z2[:,i]))/Z2[:,i] *H2_grad[:,i] , 1)], 1)

	return tf.matmul(rst, tf.transpose(W2))

H1_grad = get_H1_grad(W2, H2_grad, Z2)


# Compute gradient for W1
def get_W1_grad(x, H1_grad, Z1):
	rst = tf.expand_dims(tf.transpose(tf.reshape(tf.tile((Z1[:,0] - tf.minimum(0.0, Z1[:,0]))/Z1[:,0] * H1_grad[:,0], [784]),[784, tf.shape(pred)[0]])) * x, 2)
	for i in range(1, 100):
		rst = tf.concat([rst, tf.expand_dims(tf.transpose(tf.reshape(tf.tile((Z1[:,i] - tf.minimum(0.0, Z1[:,i]))/Z1[:,i] *H1_grad[:,i], [784]), [784, tf.shape(pred)[0]])) * x, 2)], 2)
	return rst

W1_grad = get_W1_grad(x, H1_grad, Z1)


# Compute gradient for W1_0
def get_W1_0_grad(H1_grad, Z1):
	rst = tf.expand_dims((Z1[:,0] - tf.minimum(0.0, Z1[:,0]))/Z1[:,0] * H1_grad[:,0], 1)
	for i in range(1, 100):
		rst = tf.concat([rst, tf.expand_dims((Z1[:,i] - tf.minimum(0.0, Z1[:,i]))/Z1[:,i] *H1_grad[:,i], 1)], 1)
	return rst

W1_0_grad = get_W1_0_grad(H1_grad, Z1)

# update
update_step = [tf.assign(W1, W1 - learning_rate * tf.reduce_mean(W1_grad, 0)), tf.assign(W1_0, W1_0 - learning_rate * tf.expand_dims(tf.reduce_mean(W1_0_grad, 0), 1)),  tf.assign(W2, W2 - learning_rate * tf.reduce_mean(W2_grad,  0)), tf.assign(W2_0, W2_0 - learning_rate * tf.expand_dims(tf.reduce_mean(W2_0_grad, 0), 1)), tf.assign(W3, W3 - learning_rate * tf.reduce_mean(W3_grad, 0)), tf.assign(W3_0, W3_0 - learning_rate * tf.expand_dims(tf.reduce_mean(W3_0_grad, 0), 1))]

model = tf.global_variables_initializer()

test_error = []
training_error = []
cost_value = []
theta = []

# accuracy for test
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# accuracy for training
correct_prediction2 = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))

with tf.Session() as session:
	session.run(model)
	for epoch in range(4):
		b_idx = 0
		b_size = 50
		x_value, y_value = shufflelists([x_value, y_value])
			

		while (b_idx + b_size) <= x_value.shape[0]:
			session.run(update_step, feed_dict = {x: x_value[b_idx : b_idx +b_size], y : y_value[b_idx : b_idx +b_size]})
			b_idx += b_size
				
			
			# testing error for each update
			test_error.append(1 - accuracy.eval({x: x_test, y: y_test}))
			print test_error[-1]

			# training error for each update
			training_error.append(1- accuracy2.eval({x: x_value, y: y_value}))
			print training_error[-1]

			# cost after each update
			cost_value.append(cost.eval({x: x_value, y: y_value}))
			print cost_value[-1]
			print "####"

	print ("Optimize Finished")
	# testing error after finished
	print test_error[-1]
	# training error after finished
	print training_error[-1]
	# final cost
	print cost_value[-1]

	'''
	#plot the weight
	for i in range(5):
		img = (session.run(W))[:784,i].reshape(28,28)
		plt.imshow(img)
		plt.colorbar()
		plt.show()
	'''

	#accuracy for each digit
	print ("### accuracy for each digit")
	for m in range(1, 10):
		correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_test[y_pre == m], 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		print (1 - accuracy.eval({x: x_test[y_pre == m, :], y: y_test[y_pre == m]}))
		

	theta.append(W1.eval())
	theta.append(W1_0.eval())
	theta.append(W2.eval())
	theta.append(W2_0.eval())
	theta.append(W3.eval())
	theta.append(W3_0.eval())
	filehandler = open("nn_parameters.txt","wb")
	pickle.dump(theta, filehandler, protocol = 2)
	filehandler.close()

	plot1, = plt.plot(training_error, 'b.')
	plot2, = plt.plot(test_error, 'r.')
	plt.title('training and testing error')
	plt.legend([plot1, plot2], ('training error','testing error'))
	plt.show()
	plot3, = plt.plot(cost_value, 'g.')
	plt.title('cost value')
	plt.show()


