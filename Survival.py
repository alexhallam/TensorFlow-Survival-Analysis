import tensorflow as tf
import numpy as np
import pandas as pd
# Read data
#df = pd.read_csv("simulated_dat.csv")
df = pd.read_csv("true.csv")

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
x = tf.placeholder(tf.float32, [None, 5])
y = tf.placeholder(tf.float32, [None, 1])
e = tf.placeholder(tf.float32, [None, 1])
risk_true = tf.placeholder(tf.float32, [None, 1])
#variables - initialize to vector of zeros
W = tf.Variable(tf.zeros([5,1]))
b = tf.Variable(tf.zeros([1]))

#model and cost
risk = tf.matmul(x,W) + b 
cost = -tf.reduce_mean((risk - tf.log(tf.cumsum(tf.exp(risk_true))))*e)
#Gradient Descent
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

#TensorFlow quarks
init = tf.global_variables_initializer()
sess = tf.Session()

# initialize computation graph
sess.run(init)

#Generate data
for i in range(1000):
    #features = np.array([[i]])
    features = np.array(df[['x1','x2','x3','x4','x5']])
    #target = np.array([[i*4]])
    target = np.array(df[['time']])
    censored = np.array(df[['is_censored']])
    risk_t = np.array(df[['risk']])
    #feed in data from placeholers
    feed = { x: features, y: target, e: censored, risk_true: risk_t}
    sess.run(train_step, feed_dict=feed)
    if i % 50 == 0:
        print("After %d iteration:" % i)
        print("W h(x)_1: %f" % sess.run(W[0]))
        print("W h(x)_2: %f" % sess.run(W[1]))
        print("W h(x)_3: %f" % sess.run(W[2]))
        print("W h(x)_4: %f" % sess.run(W[3]))
        print("W h(x)_5: %f" % sess.run(W[4]))
        print("b       : %f" % sess.run(b))
        print("cost    : %f" % sess.run(cost, feed_dict=feed))


