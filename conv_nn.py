import tensorflow as tf
import random
from collections import deque


class ConvNN:
    def __init__(self, l1_tot_nodes=1, ln_tot_nodes=10):
        self.nn_structure = [l1_tot_nodes, ln_tot_nodes]
    
    
    def train_nn(self, num_filters, filter_width, filter_height, 
                 pool_width, pool_height, final_grid_width, 
                 final_grid_height, test_x, test_y, train_x, train_y, 
                 learning_rate=0.0001, epochs=10, batch_size=50, 
                 img_width=28, img_height=28, stddev_w=0.03, 
                 stddev_b=0.01, pooling_stride_width=2, 
                 pooling_stride_height=2, filter_stride_width=1, 
                 filter_stride_height=1):
        # declare the training data placeholders
        # input x - for 28 x 28 pixels = 784 - this is the flattened image data that is drawn from 
        # mnist.train.nextbatch()
        x = tf.placeholder(tf.float32, [None, self.nn_structure[0]])
        # dynamically reshape the input
        # greyscale images have tot_chan=1; RGB have tot_chan=3 etc.
        # when reshaping x we do not know the tot num of training samples
        # but -1 allows to dynamically reshape as the training goes on
        tot_tr_samples = -1
        x_shaped = tf.reshape(x, [tot_tr_samples, img_height, img_width, num_filters[0]])
        # now declare the output data placeholder
        y = tf.placeholder(tf.float32, [None, self.nn_structure[0]])
        cross_entropy, y_ = self.layer_up(num_filters[0], 
                                          train_y, x_shaped, 
                                          final_grid_width, 
                                          final_grid_height, 
                                          num_filters[2], filter_width, 
                                          filter_height, pool_width, 
                                          pool_height, stddev_w, stddev_b, 
                                          pooling_stride_width, 
                                          pooling_stride_height,
                                          filter_stride_width, 
                                          filter_stride_height)
        # add an optimiser
        optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
        # define an accuracy assessment operation
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # setup the initialisation operator
        init_op = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            # initialise the variables
            sess.run(init_op)
            total_batch = int(len(train_y) / batch_size)
            for epoch in range(epochs):
                avg_cost = 0
                for i in range(total_batch):
                    batch_x = self.rand_batches(train_x, batch_size)
                    batch_y = self.rand_batches(train_y, batch_size)
                    #batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                    _, c = sess.run([optimiser, cross_entropy], 
                                    feed_dict={x: batch_x, y: batch_y})
                    avg_cost += c / total_batch
                test_acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), " test accuracy: {:.3f}".format(test_acc))
        
            print("\nTraining complete!")
            print(sess.run(accuracy, feed_dict={x: test_x, y: test_y}))
       
    
    def rand_batches(self, samples, batch_size):
        start_i = random.SystemRandom().randint(0, len(samples))
        fin = map(None, *(iter(samples),)*batch_size)
        d = deque(fin)
        d.rotate(-start_i)
        return d
    
    
    def layer_up(self, num_filters, train_y, x_shaped, final_grid_width, final_grid_height, final_chans, filter_width, filter_height, pool_width, pool_height, stddev_w, stddev_b, pooling_stride_width, pooling_stride_height, filter_stride_width, filter_stride_height):
        # create 2 convolutional layers
        layer1 = self.create_new_conv_layer(x_shaped, num_filters[0], num_filters[1], filter_width, filter_height, pool_width, pool_height, 'layer1', stddev_w, pooling_stride_width, pooling_stride_height, filter_stride_width, filter_stride_height)
        layer2 = self.create_new_conv_layer(layer1, num_filters[1], num_filters[2], filter_width, filter_height, pool_width, pool_height, 'layer2', stddev_w, pooling_stride_width, pooling_stride_height, filter_stride_width, filter_stride_height)
        # flatten out the output from the final convolutional layer.  
        # It is now a 7x7 grid of nodes with 64 channels, which equates 
        # to 3136 nodes per training sample.
        # -1 is to dynamically calculate the first dimension
        flattened = tf.reshape(layer2, [-1, final_grid_width * final_grid_height * final_chans])
        # setup some weights and bias values for this layer
        wd1 = tf.Variable(tf.truncated_normal([final_grid_width * final_grid_height * num_filters[2], 1000], stddev=stddev_w), name='wd1')
        bd1 = tf.Variable(tf.truncated_normal([1000], stddev=stddev_b), name='bd1')
        # multiply the weights with the flattened conv outut + bias
        dense_layer1 = tf.matmul(flattened, wd1) + bd1
        # apply relu activation
        dense_layer1 = tf.nn.relu(dense_layer1)
        # final layer connects to output with softmax activations
        wd2 = tf.Variable(tf.truncated_normal([1000, self.nn_structure[1]], stddev=stddev_w), name='wd2')
        bd2 = tf.Variable(tf.truncated_normal([self.nn_structure[1]], stddev=stddev_b), name='bd2')
        dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
        y_ = tf.nn.softmax(dense_layer2)
        
        # tensorflow handy function which applies soft-max followed 
        # by cross-entropy loss
        # takes the soft-max of the matrix multiplication, 
        # then compares it to the training target using cross-entropy.  
        # The result is the cross-entropy calculation per training sample
        # so we need to reduce this tensor into a scalar 
        # To do this we use tf.reduce_mean() which returns a mean of 
        # the tensor.
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=train_y))
        return cross_entropy, y_
    

    def create_new_conv_layer(self, input_data, num_input_channels, num_filters, filter_width, filter_height, pool_width, pool_height, name, stddev_w, pooling_stride_width, pooling_stride_height, filter_stride_width, filter_stride_height):
        '''
        num_filters = total amount of channels as output from the current layer 
        (if there's 1 channel as input to the 1st layer then 32 is its output (that is input to the 2nd layer)
        while 64 is the output from the 2nd layer (that is input to the 3rd layer, and so on).
        num_input_channels = total amount of channels as input to the current layer
        '''
        # setup the filter input shape for tf.nn.conv_2d
        # sets up a variable to hold the shape of the weights
        conv_filt_shape = [filter_height, filter_width, num_input_channels, num_filters]
    
        # initialise weights and bias for the filter
        weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=stddev_w), name=name+'_W')
        bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')
    
        # setup the convolutional layer operation. arguments:
        # weights: the size of the weights tensor shows TensorFlow what size the 
        # convolutional filter should be
        # [1,1,1,1]: stride parameter, i.e., 
        # [mv filter btw tr samples, step in x, step in y, mv filter btw chans]
        # padding=SAME: determines the output size of each channel and when it 
        # is set to SAME it produces dimensions of
        # out_height = ceil(float(in_height) / float(strides[1]))
        # out_width  = ceil(float(in_width) / float(strides[2]))
        out_layer = tf.nn.conv2d(input_data, weights, [1, filter_stride_width, filter_stride_height, 1], padding='SAME')
    
        # add the bias
        out_layer += bias
    
        # apply a ReLU non-linear activation
        out_layer = tf.nn.relu(out_layer)
    
        # now perform max pooling
        # first and last values are whether or not (1)
        # to apply btw tr samples or chans
        # then there's max pooling step in x and y
        ksize = [1, pool_width, pool_height, 1]
        # in order to down-sample, step in x and y of 2
        strides = [1, pooling_stride_width, pooling_stride_height, 1]
        out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')
    
        return out_layer