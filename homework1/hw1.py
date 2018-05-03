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
training_epochs = 20
regul = 0.01 

# true x_value
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/train_data/00001.jpg"
x_value = mpimg.imread(filename).reshape(1, 784)
for i in range(2, 25113):
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

x_value = np.hstack((x_value/255.0, np.ones((25112, 1))))

# true x_test
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/test_data/0001.jpg"
x_test = mpimg.imread(filename).reshape(1, 784)
for i in range(2, 4983):
	if (i >= 1) and (i <= 9):
		filename = dir_path + "/test_data/000" + str(i) + ".jpg"
	elif (i >= 10) and (i <= 99):
		filename = dir_path + "/test_data/00" + str(i) + ".jpg"
	elif (i >= 100) and (i <= 999):
		filename = dir_path + "/test_data/0" + str(i) + ".jpg"
	else:
		filename = dir_path + "/test_data/" + str(i) + ".jpg"
	x_test = np.vstack((x_test, mpimg.imread(filename).reshape(1, 784)))

x_test = np.hstack((x_test/255.0, np.ones((4982, 1))))

# true y_value
filename2 = dir_path + "/labels/train_label.txt"
y_pre = np.loadtxt(filename2)
y_value = np.zeros((25112, 5))
for i in range(25112):
	y_value[i][int(y_pre[i]) - 1] = 1


# true y_test
filename2 = dir_path + "/labels/test_label.txt"
y_pre = np.loadtxt(filename2)
y_test = np.zeros((4982, 5))
for i in range(4982):
	y_test[i][int(y_pre[i]) - 1] = 1

# use placeholder for nodes
x = tf.placeholder("float", [None, 785])
y = tf.placeholder("float", [None, 5])

# set model weights
W = tf.Variable(initial_value = tf.random_normal(shape = [785, 5], stddev = 0.01))
#W = tf.Variable(tf.zeros([785, 5]))
'''
img = tf.Session().run(W[:784,1]).reshape(28,28)
plt.imshow(int(img))
plt.colorbar()
plt.show()
'''
# Construct model
pred = tf.nn.softmax(tf.matmul(x, W))

# Cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y* tf.log(pred), reduction_indices = 1))

# gradient
W_grad = -tf.matmul(tf.transpose(x), y - pred)

# update
new_W = W.assign(W - learning_rate * (W_grad + regul* W))

model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)
	for epoch in range(20):
		b_idx = 0
		b_size = 10
		x_value, y_value = shufflelists([x_value, y_value])
		while b_idx <= x_value.shape[0]:
			session.run(new_W, feed_dict = {x: x_value[b_idx: b_idx + b_size], y: y_value[b_idx: b_idx + b_size]})
			b_idx += b_size
		
		# testing error for each epoch
		correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_test, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		print (1 - accuracy.eval({x: x_test, y: y_test}))
		print ("\n")

		#training error for each epoch
		correct_prediction2 = tf.equal(tf.argmax(pred, 1), tf.argmax(y_value, 1))
		accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
		print (1 - accuracy2.eval({x: x_value, y: y_value}))
		print ("##\n")

	#plot the weight
	for i in range(5):
		img = (session.run(W))[:784,i].reshape(28,28)
		plt.imshow(img)
		plt.colorbar()
		plt.show()

	print ("Optimize Finished\n")
#accuracy for each digit
	for m in range(1, 6):
		correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_test[y_pre == m], 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		print (1 - accuracy.eval({x: x_test[y_pre == m, :], y: y_test[y_pre == m]}))
		print ("\n")

	
	filehandler = open("multiclass_parameter.txt","wb")
	pickle.dump(session.run(W), filehandler)
	filehandler.close()
