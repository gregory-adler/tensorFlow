import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

# no limit to amount of pictures, each picture is 784 pixels
x = tf.placeholder(tf.float32, shape=[None, 784])
# no limit to amount of pictures, 10 possible options
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# pixel variable weight (what the pixel says it is likely to be)
W = tf.Variable(tf.zeros([784,10]))
# bias weight (how often the answer shows up)
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())


# regression model - 
# multiply the image input x the weight matrix (w), then add the result bias (b)
y = tf.matmul(x,W) + b