#import libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

# Next, we will load the dataset we are going to use. In this case, we are utilizing 
# the `iris` dataset, which is inbuilt -- so there's no need to do any preprocessing 
# and we can jump right into manipulating it. We separate the dataset into _xs_ and _ys_, 
# and then into training _xs_ and _ys_ and testing _xs_ and _ys_, (pseudo-)randomly.

iris = load_iris()
#take out the last row for some reason
iris_X, iris_y = iris.data[:-1,:], iris.target[:-1]
#something similar to contrasts in r. This makes one-hot vectors of the data
iris_y= pd.get_dummies(iris_y).values
# train_test_split is from the sklearn library
trainX, testX, trainY, testY = train_test_split(iris_X, iris_y, test_size=0.33, random_state=42)

# Now we define x and y. These placeholders will hold our iris data (both the features and label matrices),
# and help pass them along to different parts of the algorithm. You can consider placeholders as empty shells 
# into which we insert our data. We also need to give them shapes which correspond to the shape of our 
# data. Later, we will insert data into these placeholders by “feeding” the placeholders the data via 
# a “feed_dict” (Feed Dictionary).

# ### Why use Placeholders?  
# 1) This feature of TensorFlow allows us to create an algorithm which accepts data and knows something
# about the shape of the data without knowing the amount of data going in. 
# 2) When we insert “batches” of data in training, we can easily adjust how many examples 
# we train on in a single step without changing the entire algorithm.

# numFeatures is the number of features in our input data.
# In the iris dataset, this number is '4'.
numFeatures = trainX.shape[1]

# numLabels is the number of classes our data points can be in.
# In the iris dataset, this number is '3'.
numLabels = trainY.shape[1]

# Placeholders
# 'None' means TensorFlow shouldn't expect a fixed number in that dimension
X = tf.placeholder(tf.float32, [None, numFeatures]) # Iris has 4 features, so X is a tensor to hold our data.
yGold = tf.placeholder(tf.float32, [None, numLabels]) # This will be our correct answers matrix for 3 classes.

# ### Set model weights and bias
# 
# Much like Linear Regression, we need a shared variable weight matrix for Logistic Regression. We initialize both `W` and `b` as tensors full of zeros. Since we are going to learn `W` and `b`, their initial value doesn't matter too much. These variables are the objects which define the structure of our regression model, and we can save them after they’ve been trained so we can reuse them later.
# 
# We define two TensorFlow variables as our parameters. These variables will hold the weights and biases of our logistic regression and they will be continually updated during training. 
# 
# Notice that `W` has a shape of [4, 3] because we want to multiply the 4-dimensional input vectors by it to produce 3-dimensional vectors of evidence for the difference classes. `b` has a shape of [3] so we can add it to the output. Moreover, unlike our placeholders above which are essentially empty shells waiting to be fed data, TensorFlow variables need to be initialized with values, e.g. with zeros.

# In[ ]:

W = tf.Variable(tf.zeros([4, 3]))  # 4-dimensional input and  3 classes
b = tf.Variable(tf.zeros([3])) # 3-dimensional output [0,0,1],[0,1,0],[1,0,0]


# In[ ]:

#Randomly sample from a normal distribution with standard deviation .01

weights = tf.Variable(tf.random_normal([numFeatures,numLabels],
                                       mean=0,
                                       stddev=0.01,
                                       name="weights"))

bias = tf.Variable(tf.random_normal([1,numLabels],
                                    mean=0,
                                    stddev=0.01,
                                    name="bias"))


# ###  Logistic Regression model
# 
# We now define our operations in order to properly run the Logistic Regression. Logistic regression is typically thought of as a single equation:
# 
# $$
# ŷ =sigmoid(WX+b)
# $$
# 
# However, for the sake of clarity, we can have it broken into its three main components: 
# - a weight times features matrix multiplication operation, 
# - a summation of the weighted features and a bias term, 
# - and finally the application of a sigmoid function. 
# 
# As such, you will find these components defined as three separate operations below.
# 

# In[ ]:

# Three-component breakdown of the Logistic Regression equation.
# Note that these feed into each other.
apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias") 
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")


# As we have seen before, the function we are going to use is the _logistic function_ $(\frac{1}{1+e^{-x}})$, which is fed the input data after applying weights and bias. In TensorFlow, this function is implemented as the `nn.sigmoid` function. Effectively, this fits the weighted input with bias into a 0-100 percent curve, which is the probability function we want.

# -------------------------------------

# <a id="ref3"></a>
# # Training
# 
# The learning algorithm is how we search for the best weight vector (${\bf w}$). This search is an optimization problem looking for the hypothesis that optimizes an error/cost measure.
# 
# __What tell us our model is bad?__  
# The Cost or Loss of the model, so what we want is to minimize that. 
# 
# __What is the cost function in our model?__  
# The cost function we are going to utilize is the Squared Mean Error loss function.
# 
# __How to minimize the cost function?__   
# We can't use __least-squares linear regression__ here, so we will use [gradient descent](http://en.wikipedia.org/wiki/Gradient_descent) instead. Specifically, we will use batch gradient descent which calculates the gradient from all data points in the data set.
# 
# ### Cost function
# Before defining our cost function, we need to define how long we are going to train and how should we define the learning rate.

# In[ ]:

# Number of Epochs in our training
numEpochs = 700

# Defining our learning rate iterations (decay)
learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                          global_step= 1,
                                          decay_steps=trainX.shape[0],
                                          decay_rate= 0.95,
                                          staircase=True)


# In[ ]:

#Defining our cost function - Squared Mean Error
cost_OP = tf.nn.l2_loss(activation_OP-yGold, name="squared_error_cost")

#Defining our Gradient Descent
training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)


# Now we move on to actually running our operations. We will start with the operations involved in the prediction phase (i.e. the logistic regression itself).
# 
# First, we need to initialize our weights and biases with zeros or random values via the inbuilt Initialization Op, __tf.initialize_all_variables()__. This Initialization Op will become a node in our computational graph, and when we put the graph into a session, then the Op will run and create the variables.

# In[ ]:

# Create a tensorflow session
sess = tf.Session()

# Initialize our weights and biases variables.
init_OP = tf.global_variables_initializer()

# Initialize all tensorflow variables
sess.run(init_OP)


# We also want some additional operations to keep track of our model's efficiency over time. We can do this like so:

# In[ ]:

# argmax(activation_OP, 1) returns the label with the most probability
# argmax(yGold, 1) is the correct label
correct_predictions_OP = tf.equal(tf.argmax(activation_OP,1),tf.argmax(yGold,1))

# If every false prediction is 0 and every true prediction is 1, the average returns us the accuracy
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))

# Summary op for regression output
activation_summary_OP = tf.summary.histogram("output", activation_OP)

# Summary op for accuracy
accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)

# Summary op for cost
cost_summary_OP = tf.summary.scalar("cost", cost_OP)

# Summary ops to check how variables (W, b) are updating after each iteration
weightSummary = tf.summary.histogram("weights", weights.eval(session=sess))
biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))

# Merge all summaries
merged = tf.summary.merge([activation_summary_OP, accuracy_summary_OP, cost_summary_OP, weightSummary, biasSummary])

# Summary writer
writer = tf.summary.FileWriter("summary_logs", sess.graph)


# Now we can define and run the actual training loop, like this:

# In[ ]:

# Initialize reporting variables
cost = 0
diff = 1
epoch_values = []
accuracy_values = []
cost_values = []

# Training epochs
for i in range(numEpochs):
    if i > 1 and diff < .0001:
        print("change in cost %g; convergence."%diff)
        break
    else:
        # Run training step
        step = sess.run(training_OP, feed_dict={X: trainX, yGold: trainY})
        # Report occasional stats
        if i % 10 == 0:
            # Add epoch to epoch_values
            epoch_values.append(i)
            # Generate accuracy stats on test data
            train_accuracy, newCost = sess.run([accuracy_OP, cost_OP], feed_dict={X: trainX, yGold: trainY})
            # Add accuracy to live graphing variable
            accuracy_values.append(train_accuracy)
            # Add cost to live graphing variable
            cost_values.append(newCost)
            # Re-assign values for variables
            diff = abs(newCost - cost)
            cost = newCost

            #generate print statements
            print("step %d, training accuracy %g, cost %g, change in cost %g"%(i, train_accuracy, newCost, diff))


# How well do we perform on held-out test data?
print("final accuracy on test set: %s" %str(sess.run(accuracy_OP, 
                                                     feed_dict={X: testX, 
                                                                yGold: testY})))

#import numpy as np
#import matplotlib.pyplot as plt
#plt.plot([np.mean(cost_values[i-50:i]) for i in range(len(cost_values))])
#plt.show()

# Assuming no parameters were changed, you should reach a peak accuracy of 90% at the end of 
# training, which is commendable. Try changing the parameters such as the length of training, 
# and maybe some operations to see how the model behaves. Does it take much longer? How is the performance?

np.savetxt("cost_values.csv", cost_values, delimiter=",")
