import pandas as pd
import numpy as np
import tensorflow as tf
import time 

ADAM_OPTIMIZER_PARAM = 1e-2
CNN_MAX_ITERATION = 100#int(raw_input("GRAD_DESC_MAX_ITERATION = ")) 
TRAIN_TEST_ALLOC = 1500 
PIX = 96
BATCH_SIZE = 50

if BATCH_SIZE > 5*TRAIN_TEST_ALLOC:
    BATCH_SIZE = 5*TRAIN_TEST_ALLOC

start_time = time.ctime() 
print "Let's begin cnn. " + start_time + "\n"
if PIX == 96:
    kaishu_name = '/Users/Lee/Documents/Courses/CS229/CS229 Project/data_96_bw/kaishu_96_bw.csv'
    caoshu_name = '/Users/Lee/Documents/Courses/CS229/CS229 Project/data_96_bw/caoshu_96_bw.csv'
    lishu_name = '/Users/Lee/Documents/Courses/CS229/CS229 Project/data_96_bw/lishu_96_bw.csv'
    xingshu_name = '/Users/Lee/Documents/Courses/CS229/CS229 Project/data_96_bw/xingshu_96_bw.csv'
    zhuanshu_name = '/Users/Lee/Documents/Courses/CS229/CS229 Project/data_96_bw/zhuanshu_96_bw.csv'
elif PIX == 28:
	kaishu_name = '/Users/Lee/Documents/Courses/CS229/CS229 Project/data_28_bw/kaishu_28_bw.csv'
	caoshu_name = '/Users/Lee/Documents/Courses/CS229/CS229 Project/data_28_bw/caoshu_28_bw.csv' 
	lishu_name = '/Users/Lee/Documents/Courses/CS229/CS229 Project/data_28_bw/lishu_28_bw.csv' 
	xingshu_name = '/Users/Lee/Documents/Courses/CS229/CS229 Project/data_28_bw/xingshu_28_bw.csv' 
	zhuanshu_name = '/Users/Lee/Documents/Courses/CS229/CS229 Project/data_28_bw/zhuanshu_28_bw.csv' 

print "Start importing training and testing data. " + time.ctime()
kaishu = pd.read_csv(kaishu_name, header = None)
print "Finished: importing kaishu. " + time.ctime()
caoshu = pd.read_csv(caoshu_name, header = None)
print "Finished: importing caoshu. " + time.ctime()
lishu = pd.read_csv(lishu_name, header = None)
print "Finished: importing lishu. " + time.ctime()
xingshu = pd.read_csv(xingshu_name, header = None)
print "Finished: importing xingshu. " + time.ctime()
zhuanshu = pd.read_csv(zhuanshu_name, header = None)
print "Finished: importing zhuanshu. " + time.ctime()
print "Finished importing training data. " + time.ctime() + "\n"

kaishu_train = np.array(kaishu[:TRAIN_TEST_ALLOC]) + 6 # 21
caoshu_train = np.array(caoshu[:TRAIN_TEST_ALLOC]) # 27
lishu_train = np.array(lishu[:TRAIN_TEST_ALLOC]) + 5 # 22
xingshu_train = np.array(xingshu[:TRAIN_TEST_ALLOC]) + 1 # 26
zhuanshu_train = np.array(zhuanshu[:TRAIN_TEST_ALLOC]) # 27

kaishu_test = np.array(kaishu[TRAIN_TEST_ALLOC:]) + 6
caoshu_test = np.array(caoshu[TRAIN_TEST_ALLOC:])
lishu_test = np.array(lishu[TRAIN_TEST_ALLOC:]) + 5
xingshu_test = np.array(xingshu[TRAIN_TEST_ALLOC:]) + 1
zhuanshu_test = np.array(zhuanshu[TRAIN_TEST_ALLOC:])

kaishu_train_label = np.array([[1,0,0,0,0]])[[0]*len(kaishu_train)]
caoshu_train_label = np.array([[0,1,0,0,0]])[[0]*len(caoshu_train)]
lishu_train_label = np.array([[0,0,1,0,0]])[[0]*len(lishu_train)]
xingshu_train_label = np.array([[0,0,0,1,0]])[[0]*len(xingshu_train)]
zhuanshu_train_label = np.array([[0,0,0,0,1]])[[0]*len(zhuanshu_train)]

kaishu_test_label = np.array([[1,0,0,0,0]])[[0]*len(kaishu_test)]
caoshu_test_label = np.array([[0,1,0,0,0]])[[0]*len(caoshu_test)]
lishu_test_label = np.array([[0,0,1,0,0]])[[0]*len(lishu_test)]
xingshu_test_label = np.array([[0,0,0,1,0]])[[0]*len(xingshu_test)]
zhuanshu_test_label = np.array([[0,0,0,0,1]])[[0]*len(zhuanshu_test)]

train_x = np.concatenate((kaishu_train, caoshu_train, lishu_train, xingshu_train, zhuanshu_train), axis=0)
train_y = np.concatenate((kaishu_train_label, caoshu_train_label, lishu_train_label, xingshu_train_label, zhuanshu_train_label), axis=0)

test_x = np.concatenate((kaishu_test, caoshu_test, lishu_test, xingshu_test, zhuanshu_test), axis=0)
test_y = np.concatenate((kaishu_test_label, caoshu_test_label, lishu_test_label, xingshu_test_label, zhuanshu_test_label), axis=0)

del kaishu
del caoshu
del lishu
del xingshu
del zhuanshu

del kaishu_train 
del caoshu_train 
del lishu_train 
del xingshu_train 
del zhuanshu_train

del kaishu_test 
del caoshu_test 
del lishu_test 
del xingshu_test 
del zhuanshu_test 

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, PIX**2])
W = tf.Variable(tf.zeros([PIX**2, 5]))
b = tf.Variable(tf.zeros([5]))
y = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, 5])

# cnn
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_1x1(x):
	return tf.nn.max_pool(x, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,PIX,PIX,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 32, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 32, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([12 * 12 * 128, 128])
b_fc1 = bias_variable([128])
h_pool3_flat = tf.reshape(h_pool3, [-1, 12 * 12 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([128, 5])
b_fc2 = bias_variable([5])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(ADAM_OPTIMIZER_PARAM).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for iter in range(CNN_MAX_ITERATION):
	if BATCH_SIZE == 5*TRAIN_TEST_ALLOC:
	    batch_ind = range(5*TRAIN_TEST_ALLOC) 
	else:
	    batch_ind = np.random.randint(low=0,high=len(train_x), size=BATCH_SIZE)
	batch_xs = train_x[batch_ind]
	batch_ys = train_y[batch_ind]
	if iter % (CNN_MAX_ITERATION/20) == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
		print("step %d, training accuracy %g"%(iter, train_accuracy))
		test_accuracy = accuracy.eval(feed_dict={x: test_x, y_: test_y, keep_prob: 1.0})
		print("step %d, test accuracy %g"%(iter, test_accuracy)) + "\n"
	train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.7})

print "Finshed running cnn learning algorithm. " + time.ctime() + "\n"

print "\nTraining accuracy:"
train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
print("%g"%train_accuracy)
print "Testing accuracy:"
test_accuracy = accuracy.eval(feed_dict={x: test_x, y_: test_y, keep_prob: 1.0})
print("%g"%test_accuracy)
end_time = time.ctime() 
print "\n start time: " + start_time + " end time: " + end_time 
