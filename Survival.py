import tensorflow as tf
import numpy as np
import pandas as pd
# Read data
df = pd.read_csv("simulated_dat.csv")

# 1. Identify model and cost
# Linear Regression Model y_hat = Wx+b
# Cost = \sum_{i \in D}[F(x_i,\theta) - log(\sum_{j \in R_i} e^F(x_j,\theta))] - \lambda P(\theta)

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
e = tf.placeholder(tf.float32, [None, 1])
#variables - initialize to vector of zeros
W = tf.Variable(tf.zeros([1,1]))
#b = tf.Variable(tf.zeros([1]))

#model and cost
risk = tf.matmul(x,W)
cost = -tf.reduce_sum((risk - tf.log(tf.cumsum(tf.exp(risk))))*e)
#Gradient Descent
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

#TensorFlow quarks
init = tf.global_variables_initializer()
sess = tf.Session()

# initialize computation graph
sess.run(init)

#Generate data
for i in range(500):
    #features = np.array([[i]])
    features = np.array(df[['x']])
    #target = np.array([[i*4]])
    target = np.array(df[['t']])
    censored = np.array(df[['e']])
    #feed in data from placeholers
    feed = { x: features, y: target, e: censored }
    sess.run(train_step, feed_dict=feed)
    if i % 50 == 0:
        print("After %d iteration:" % i)
        print("W: %f" % sess.run(W))
        print("cost: %f" % sess.run(cost, feed_dict=feed))


