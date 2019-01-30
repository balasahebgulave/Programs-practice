# https://github.com/vahidk/EffectiveTensorflow#basics
# coding: utf-8


# ## Tensorflow
# For CPU

pip install tensorflow

# For GPU
# Other libraries 

pip install tensorflow-gpu
pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
pip install --user jupyter
pip install --user matplotlib

# In[1]:


# Hello World program with tensorflow

import tensorflow as tf
a = tf.constant('Hello Tensorflow !')
b = tf.constant('Welcome to Tensorflow Tutorial.')
session = tf.Session()
a = session.run(a)
b = session.run(b)
print(a)
print(b)
session.close()


# In[2]:


# tensorflow constants objective - value which cannot change

import tensorflow as tf

a = tf.constant(5.0)
b = tf.constant(6.0)

c = a*b

session = tf.Session()
c = session.run(c)
print(c)
session.close()


# In[3]:


# tensorflow placeholders objective - To get data from outside

import tensorflow as tf 

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

c = a*b

session = tf.Session()
c = session.run(c , {a:[1,3],b:[2,4]})
print(c)
session.close()


# In[4]:


# tensorflow variables - 

import tensorflow as tf
a = tf.Variable([0.4], dtype = tf.float32)
b = tf.Variable([0.4], dtype = tf.float32)
c = a*b
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
print(session.run(c))
session.close()

# tensorflow program does not execute without session.
# tensorflow is different way of programming than traditional programming .
# which run the code with computational graph and in tensorflow session .
# session is run to evaluate nodes. This is called as tensorflow runtime .
# In[5]:


# Basic program to explain the concept of constant ,variable and placeholder in tensorflow.
# While running the expression using variable we need to initializze the tensorflow global global_variable_initializer

import tensorflow as tf
a = tf.Variable(1)
b = tf.constant(1)
c = tf.add(a,b)
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
c = session.run(c)
print(c)
session.close()


# In[6]:


# updating or changing values of variable.
# we cant change the values of constant.

import tensorflow as tf

a = tf.Variable(10)
a = tf.assign(a,6)
session = tf.Session()
c = session.run(a)
for i in range(c):
    print(i)
session.close()


# In[7]:


# tensorflow operation with string.

import tensorflow as tf

a = tf.constant('Hello ')
b = tf.constant('World !')
c = tf.add(a,b)
session = tf.Session()
c = session.run(c)
print(c)
session.close()


# In[8]:


# placeholder in tensorflow and how to deal with it.

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = a*b
session = tf.Session()
c = session.run(c , {a:[1,2,3],b:[4,5,6]})
print(c)
session.close()


# In[47]:


import tensorflow as tf
graph = tf.get_default_graph()
a = tf.constant(10 , name = 'a')
operations = graph.get_operations()


# In[48]:


b = tf.constant(20 , name = 'b')
operations = graph.get_operations()


# In[11]:


c = a*b
session = tf.Session()
c = session.run(c)
print(c)


# In[50]:


for i in graph.get_operations():
    pass
session.close()


# In[51]:


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = a * b
session = tf.Session()
session.run(c , feed_dict = {a:[1,2,3,4,5],b:[6,7,8,9,10]})


# # Model Building

# Four steps to be consider to built your model
# 
# 1. If you want to train your model you need "training data".
# 
# 2. Model -
# 
# 3. Cost function -
# 
#         a. For Regression model - technique :- least_square_error()
#         e.g - cost = tf.reduce_mean(tf.square(y_pred - y_true))
#         
#         b. For Classification model -technique :- softmax_cross_entropy_with_logits()
#         e.g - cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=yhat))
# 
# 4. Optimization -
# 
#         a. AdamOptimizer
#         e.g - optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#         
#         b. GradientDescentOptimizer
#         e.g - optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) 
#         
# 5. Evaluation criteria - 
# 
#      a. training_accuracy
#      e.g - train_accuracy = np.mean(np.argmax(y_train, axis=1) == sess.run(ypredict, feed_dict={X : X_train, y: y_train}))
#      
#      b. testing_accuracy
#      e.g - test_accuracy  = np.mean(np.argmax(y_test, axis=1) == sess.run(ypredict, feed_dict={X : X_test, y: y_test}))
# 

# # Model Example

# In[18]:


# http://www.insightsbot.com/blog/2CrCd3/tensorflow-tutorial-iris-classification-with-sgd
import pandas as pd
import numpy as np
import requests
import re
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from pandas import get_dummies
from sklearn import datasets


# In[19]:


dataset = pd.read_csv('Iris_data/iris.csv')


# In[20]:


visualizations = sns.pairplot(dataset , hue = 'Species' , size = 2 )


# In[5]:


from sklearn.preprocessing import LabelBinarizer

species_lb = LabelBinarizer()
Y = species_lb.fit_transform(dataset.Species.values)


# In[6]:


from sklearn.preprocessing import normalize
FEATURES = dataset.columns[0:4]
X_data = dataset[FEATURES].as_matrix()
X_data = normalize(X_data)


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.3, random_state=1)


# In[8]:


import tensorflow as tf

# Parameters
learning_rate = 0.01
training_epochs = 120


# In[9]:


# Neural Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 128 # 1st layer number of neurons
n_input = X_train.shape[1] # input shape (105, 4)
n_classes = y_train.shape[1] # classes to predict


# In[10]:


# Inputs
X = tf.placeholder("float", shape=[None, n_input])
y = tf.placeholder("float", shape=[None, n_classes])

# Dictionary of Weights and Biases
weights = {
  'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
  'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
  'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
  'b1': tf.Variable(tf.random_normal([n_hidden_1])),
  'b2': tf.Variable(tf.random_normal([n_hidden_2])),
  'out': tf.Variable(tf.random_normal([n_classes]))
}


# In[11]:


# Model Forward Propagation step
def forward_propagation(x):
    # Hidden layer1
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Output fully connected layer
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out'] 
    return out_layer


# Model Outputs
yhat = forward_propagation(X)
ypredict = tf.argmax(yhat, axis=1)


# In[12]:


# Backward propagation
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)


train_op = optimizer.minimize(cost)


# In[13]:


# Initializing the variables
init = tf.global_variables_initializer()

from datetime import datetime
startTime = datetime.now()

# with tf.Session() as sess:
sess = tf.Session()
sess.run(init)

#writer.add_graph(sess.graph)
#EPOCHS
for epoch in range(training_epochs):
    #Stochasting Gradient Descent
    for i in range(len(X_train)):
        summary = sess.run(train_op, feed_dict={X: X_train[i: i + 1], y: y_train[i: i + 1]})

    train_accuracy = np.mean(np.argmax(y_train, axis=1) == sess.run(ypredict, feed_dict={X: X_train, y: y_train}))
    test_accuracy  = np.mean(np.argmax(y_test, axis=1) == sess.run(ypredict, feed_dict={X: X_test, y: y_test}))

    print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
    #print("Epoch = %d, train accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy))

# sess.close()
print("Time taken:", datetime.now() - startTime)


# In[27]:


# Model Outputs
yhat = forward_propagation(X)
ypredict = tf.argmax(yhat, axis=1)

# give your input here for model prediction
real_input = [4,5,6,7]
real_input = np.array([real_input])
real_input = normalize(real_input)
model_output = sess.run(ypredict , {X : real_input})


# In[28]:


labels = species_lb.inverse_transform(Y)
result = labels[model_output]
result

