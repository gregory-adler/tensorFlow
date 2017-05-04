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


# loss function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))