import tensorflow as tf
import numpy as np
# 1. Identify model and cost
# Linear Regression Model y_hat = Wx+b
# Cost sum((y-y_hat)^2)

# 2. Identify placeholders
# x
# y

# 3. Identify Variables
# W and b are variables
# Everything else -- components that are not a variables or placeholder -- 
# should be a combination of these building blocks


#placeholders
# None means that we don't want to specify the number of rows
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

#variables - initialize to vector of zeros
W = tf.Variable(tf.zeros([1,1]))
b = tf.Variable(tf.zeros([1]))

#model and cost
y_hat = tf.matmul(x,W) + b
cost = tf.reduce_sum(tf.pow(y-y_hat,2))

#Gradient Descent
train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)

#TensorFlow quarks
init = tf.global_variables_initializer()
sess = tf.Session()

# initialize computation graph
sess.run(init)

#Generate data
for i in range(100):
    features = np.array([[i]])
    target = np.array([[i*4]])
    #feed in data from placeholers
    feed = { x: features, y: target }
    sess.run(train_step, feed_dict=feed)
    print("After %d iteration:" % i)
    print("W: %f" % sess.run(W))
    print("b: %f" % sess.run(b))
    print("cost: %f" % sess.run(cost, feed_dict=feed))


