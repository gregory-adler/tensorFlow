import tensorflow as tf

#constants
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) 

# doesn't print constant because its only during a session
print(node1, node2)

# prints nodes
sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))


# placeholders
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b 


# how to make use of a placeholder with parameters
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))


# Variables
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)

# model with error
y = tf.placeholder(tf.float32)
linear_model = W * x + b
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# Constants are initialized when you call tf.constant, and their value can never change. 
# By contrast, variables are not initialized when you call tf.Variable. 
# To initialize all the variables in a TensorFlow program, you must explicitly call a special operation as follows:
init = tf.global_variables_initializer()
sess.run(init)


# gradient optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  # x is inputs, y is expected outputs
  # is modifyign the variables w and b
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W, b]))