import tensorflow as tf
import random
from collections import deque

class ThreeLayerNN:
    def __init__(self, l1_tot_nodes=0, l2_tot_nodes=0, l3_tot_nodes=0):
        self.nn_structure = [l1_tot_nodes, l2_tot_nodes, l3_tot_nodes]
    
    def init_x_y_placeholders(self):
        # training data placeholders
        # e.g., 49 features per packet --> l1_tot_nodes = 49 --> array with 49 dimensions, accepting float and None elems
        x = tf.placeholder(tf.float32, [None, self.nn_structure[0]])
        # output data placeholder
        y = tf.placeholder(tf.float32, [None, self.nn_structure[2]])
        return x, y
    
    
    def init_rand_norm_weights_biases(self, stddev, nn_struct):
        # init (tot_layers-1 = 2) weights and biases using
        # a random normal distribution 
        # with a mean of zero and a given standard deviation 
        tot = len(nn_struct)-1
        w = [tot]
        b = [tot]
        for i in range(0, tot-1):
            w[i] = tf.Variable(tf.random_normal([nn_struct[i], self.nn_structure[i+1]], stddev), name='W'+str(i))
            b[i] = tf.Variable(tf.random_normal([nn_struct[i]]), name='b'+str(i))
        return w, b
    
    
    def train_nn(self, inputs_x, outputs_y, w, b, max_iterations = 3000, step = 0.5, epochs = 10, batch_size = 100, stddev=0.03):
        x, y = self.init_x_y_placeholders()
        # now declare the weights connecting the input to the hidden layer
        W1 = tf.Variable(tf.random_normal([43, 20], stddev), name='W1')
        b1 = tf.Variable(tf.random_normal([20]), name='b1')
        # and the weights connecting the hidden layer to the output layer
        W2 = tf.Variable(tf.random_normal([20, 10], stddev), name='W2')
        b2 = tf.Variable(tf.random_normal([10]), name='b2')
        # HIDDEN LAYER
        # matrix multiplication btw inputs and weights + bias
        print tf.size(x, None, tf.int32)
        print tf.size(W1, None, tf.int32)
        mul = tf.matmul(x, W1)
        print mul
        hidden_out = tf.add(mul, b1)
        # get the final hidden layer output by applying the ReLU activation function
        hidden_out = tf.nn.relu(hidden_out)
        
        # OUTPUT LAYER
        # matrix multiplication btw inputs and weights + bias
        final_out = tf.add(tf.matmul(hidden_out, W2), b2)
        # get the final hidden layer output by applying the softMax activation function
        final_out = tf.nn.softmax(final_out)
        
        # "clip" the final layer output, i.e., limit values btw le-10 and 0.9999
        # in order to ensure no log(0) during cross entropy calc are going to hinder training process
        final_out_clipped = tf.clip_by_value(final_out, 1e-10, 0.9999999)
        # cost (aka loss) function needed by the backpropagation step
        #
        # final_out and final_out_clipped are mx10 tensors
        # therefore in cross-entropy we are interested in the second axis
        # of the tensor (i.e., the one with 10elems) which corresp. to axis "1"
        # reduce_sum returns sum of a given axis of the supplied tensor
        # which results in a mx1 tensor
        # reduce_mean returns the mean of the mx1 tensor
        cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(final_out_clipped) + (1 - y) * tf.log(1 - final_out_clipped), axis=1))
        
        # setup optimisation and backpropagation based on cross_entropy minimisation
        optimiser = tf.train.GradientDescentOptimizer(learning_rate=step).minimize(cross_entropy)

        # setup the initialisation operator
        init_op = tf.global_variables_initializer()
        
        # define an accuracy assessment operation
        # argmax returns index of max value in tensor
        # correct_prediction returns mx1 tensor of true/false elems 
        # corresp to whether the elem was predicted correctly or not
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(final_out, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracies = []
        costs = []
        # start the session
        with tf.Session() as sess:
            # initialise the variables
            sess.run(init_op)
            batches_x = self.rand_batches(inputs_x, batch_size)
            batches_y = self.rand_batches(outputs_y, batch_size)
            # calculate the number of batches to run through in each training epoch
            # total_batch = int(len(mnist.train.labels) / batch_size)
            total_batch = len(batches_y)
            for epoch in range(epochs):
                # keep track of the average cross entropy cost for each epoch
                avg_cost = 0
                for i in range(total_batch):
                    # extract a randomised batch of samples
                    # batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                    batch_x = batches_x[i]
                    batch_y = batches_y[i]
                    # run first optimisation then cross_entropy on the rand samples 
                    # and collect their outputs 
                    _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                    avg_cost += c / total_batch
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
                costs.append(avg_cost)
            accuracies.append(sess.run(accuracy, feed_dict={x: inputs_x, y: outputs_y})) 
        return accuracies, costs
    
    
    def rand_batches(self, samples, batch_size):
        start_i = random.SystemRandom().randint(0, len(samples))
        fin = map(None, *(iter(samples),)*batch_size)
        d = deque(fin)
        d.rotate(-start_i)
        return d

   