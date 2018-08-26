import tensorflow as tf
import numpy as np

# tensorflow constant and variables
const = tf.constant(4.0, name='mycon')
b = tf.Variable(4.0, name='b')
c = tf.Variable(3.0, name='c')

sum_bc = tf.add(b, c, name='sum_bc')
sum_ccon = tf.add(c, const, name='sum_ccon')
mul_sums = tf.multiply(sum_bc, sum_ccon, name='mul_sums')

initialiser = tf.global_variables_initializer()

# tensorflow session = obj where all ops run
with tf.Session() as session:
    # initialise the vars
    session.run(initialiser)
    # get output of computational graph
    # thanks to the tensorflow data flow graph, we need to call only mul_sums
    # and all ops that were supposed to happen before in order to obtain the 
    # actual values for the multiplication operands, get automatically triggered
    graph_out = session.run(mul_sums)
    print ("Variable mul_sums is {}".format(graph_out)) 
    
# if we had a var for which we didn't know the value
# before the tf.Session() began, then we would use a tf placeholder
# e.g., here the var is a 1-dim array which accepts float elems and also None
unknown = tf.placeholder(tf.float32, [None, 1], name='unknown')
with tf.Session() as s:
    # run the initialiser for the known variable
    s.run(initialiser)
    g_out = s.run(mul_sums, feed_dict={unknown: np.arange(0, 10)[:, np.newaxis]})
    print ("Variable mul_sums is {}".format(g_out)) 
