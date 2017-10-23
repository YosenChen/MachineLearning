import pandas as pd
import numpy as np
import tensorflow as tf
import time 

print "Let's begin!\n" + time.ctime() 
kaishu_name = 'kaishu.csv'
caoshu_name = 'caoshu.csv'
lishu_name = 'lishu.csv'
xingshu_name = 'xingshu.csv'
zhuanshu_name = 'zhuanshu.csv'

print "Start importing training data.\n" + time.ctime()
kaishu = pd.read_csv(kaishu_name, header = None)
print "Finished: importing kaishu.\n" + time.ctime()
caoshu = pd.read_csv(caoshu_name, header = None)
print "Finished: importing caoshu.\n" + time.ctime()
lishu = pd.read_csv(lishu_name, header = None)
print "Finished: importing lishu.\n" + time.ctime()
xingshu = pd.read_csv(xingshu_name, header = None)
print "Finished: importing xingshu.\n" + time.ctime()
zhuanshu = pd.read_csv(zhuanshu_name, header = None)
print "Finished: importing zhuanshu.\n" + time.ctime()
print "Finished importing training data.\n" + time.ctime()

kaishu_train = np.array(kaishu[:1500]) + 6 # 21
caoshu_train = np.array(caoshu[:1500]) # 27
lishu_train = np.array(lishu[:1500]) + 5 # 22
xingshu_train = np.array(xingshu[:1500]) + 1 # 26
zhuanshu_train = np.array(zhuanshu[:1500]) # 27

kaishu_test = np.array(kaishu[1500:]) + 6
caoshu_test = np.array(caoshu[1500:])
lishu_test = np.array(lishu[1500:]) + 5
xingshu_test = np.array(xingshu[1500:]) + 1
zhuanshu_test = np.array(zhuanshu[1500:])

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

print "Starting running softmax learning algorithm.\n" + time.ctime()
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 5]))
b = tf.Variable(tf.zeros([5]))
y = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, 5])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)



# Train
tf.initialize_all_variables().run()
for _ in range(1000):
	batch_ind = np.random.randint(low=0,high=len(train_x), size=100)
	batch_xs = train_x[batch_ind]
	batch_ys = train_y[batch_ind]
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print "Finshed running softmax learning algorithm.\n" + time.ctime()

print "Starting testing.\n" + time.ctime()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print "Finished testing.\n" + time.ctime()

print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
