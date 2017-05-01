import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) 
# doesn't print constant because its only during a session
print(node1, node2)

sess = tf.Session()
# prints nodes
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)


# how to make use of a placeholder with parameters
print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))


# Constants
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# Constants are initialized when you call tf.constant, and their value can never change. 
# By contrast, variables are not initialized when you call tf.Variable. 
# To initialize all the variables in a TensorFlow program, you must explicitly call a special operation as follows:

init = tf.global_variables_initializer()
sess.run(init)
