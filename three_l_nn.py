import numpy as np
import numpy.random as nprand


class ThreeLayerNN:
    def __init__(self, l1_tot_nodes=0, l2_tot_nodes=0, l3_tot_nodes=0):
        self.nn_structure = [l1_tot_nodes, l2_tot_nodes, l3_tot_nodes]
    

    def sigmoid(self, input_x):
        """
        Sigmoid Function is used here as the node activation function.
        """
        return 1 / (1 + np.exp(-input_x))
    
    
    def sigmoid_derivative(self, input_x):
        return self.sigmoid(input_x) * (1 - self.sigmoid(input_x))
    
    
    def init_weights_biases(self):
        """
        Step 1.
        Randomly initialise weights and biases for input layer
        and all hidden layers.
        """
        weights = {}
        biases = {}
        for layer in range(1, len(self.nn_structure)):
            weights[layer] = nprand.random_sample((self.nn_structure[layer], self.nn_structure[layer-1]))
            biases[layer] = nprand.random_sample((self.nn_structure[layer],))
        return weights, biases
    
    
    def reset_cost_partial_derivs_sum(self):
        """
        Step 2.
        At each iteration until the exit condition is reached,
        set the cumulative sum of all partial derivatives
        of the cost function, to zero
        for the input layer and all hidden layer's weights and biases
        """
        cost_weights_partial_derivs_sums = {}
        cost_biases_partial_derivs_sums = {}
        for layer in range(1, len(self.nn_structure)):
            cost_weights_partial_derivs_sums[layer] = np.zeros((self.nn_structure[layer], self.nn_structure[layer-1]))
            cost_biases_partial_derivs_sums[layer] = np.zeros((self.nn_structure[layer],))
        return cost_weights_partial_derivs_sums, cost_biases_partial_derivs_sums
        
        
    def feed_forward(self, inputs_x, weights, biases):
        """
        Step 3.
        For each sample,
        perform a feed-forward pass through all layers
        """
        # the input to the first layer nodes are the Xs (i.e., features
        # of the current input sample
        layer_to_activation_funct_outputs = {1: inputs_x}
        z = {}
        for layer in range(1, len(weights)+1):
            if layer == 1:
                nodes_input = inputs_x
            else:
                nodes_input = layer_to_activation_funct_outputs[layer]
            z[layer+1] = weights[layer].dot(nodes_input) + biases[layer]
            layer_to_activation_funct_outputs[layer+1] = self.sigmoid(z[layer+1])
        return layer_to_activation_funct_outputs, z
    
    
    def output_layer_cost_gradient(self, expected_outputs, actual_activation_outputs, actual_outputs):
        """
        Step 4.
        At the end of the feed-forward pass on the output layer,
        if the minimum of the cost function curve has not been reached yet
        then Calculate the gradient (slope) of the cost function 
        in order to later decide how to better adjust weights and biases.
        """
        # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
        return -(expected_outputs - actual_activation_outputs) * self.sigmoid_derivative(actual_outputs)
    
    
    def hidden_layers_cost_gradient(self, gradient_at_layer_plus_1, weights_at_layer, z_at_layer):
        """
        Step 5.
        Use back-propagation to calculate the gradient of the cost function
        for all hidden layers
        """
        return np.dot(np.transpose(weights_at_layer), gradient_at_layer_plus_1) * self.sigmoid_derivative(z_at_layer)
    
    
    def convert_single_to_multi_output(self, expected_class):
        expected_outputs = []
        for ind in range(0, self.nn_structure[2]):
            if ind == expected_class:
                expected_outputs.append(1)
            else:
                expected_outputs.append(0)
        return expected_outputs
    
    
    def convert_multi_to_single_output(self, multi_output):
        for ind in range(0, len(multi_output)):
            if multi_output[ind] != 0:
                return ind
        return -1
    

    def train_nn(self, inputs_x, outputs_y, weights, biases, max_iterations = 3000, step = 0.25):
        # for each sample, iterate feed-fwd and back-propagation to adjust the weights and biases
        avg_cost_curve = []
        for sample_index in range(0, len(inputs_x)):
            print('Training NN with sample #{}').format(sample_index)
            expected_outputs = self.convert_single_to_multi_output(outputs_y[sample_index])            
            inputs_x_curr = inputs_x[sample_index]
            iteration = 0
            tot_classes_to_predict = self.nn_structure[2]
            output_layer = len(self.nn_structure)
            print('Starting gradient descent for {} iterations').format(max_iterations)
            # at each iteration until the limit is reached
            while iteration < max_iterations:
                if iteration % 1000 == 0:
                    print('Iteration {} of {}'.format(iteration, max_iterations))
                # 2) reset the cumulative sum of partial derivatives for weights and for biases to zero
                cost_weights_partial_derivs_sums, cost_biases_partial_derivs_sums = self.reset_cost_partial_derivs_sum() 
                avg_cost = 0
                # for each feature (i.e., input x) of the current sample
                for predicted_class_index in range(len(expected_outputs)):
                    cost_gradients = {}
                    # 3) feed-fwd through all layers
                    layer_to_activation_funct_output, z = self.feed_forward(inputs_x_curr, weights, biases)
                    # 4) 5) back-propagation: calculate cost, its gradient and consequent adjustments to weights and biases
                    for layer in range(output_layer, 0, -1):
                        if layer == output_layer:
                            cost_gradients[layer] = self.output_layer_cost_gradient(expected_outputs[predicted_class_index], layer_to_activation_funct_output[layer], z[layer])
                            avg_cost += np.linalg.norm((expected_outputs[predicted_class_index] - layer_to_activation_funct_output[layer]))
                        else:
                            if layer > 1:
                                cost_gradients[layer] = self.hidden_layers_cost_gradient(cost_gradients[layer+1], weights[layer], z[layer])
                            cost_weights_partial_derivs_sums[layer] += np.dot(cost_gradients[layer+1][:, np.newaxis], np.transpose(layer_to_activation_funct_output[layer][:, np.newaxis]))
                            cost_biases_partial_derivs_sums[layer] += cost_gradients[layer+1]
                # 6) adjust weights and biases accordingly (i.e., gradient descent step)
                for layer in range(output_layer -1, 0, -1):
                    weights[layer] += -step * (1.0/tot_classes_to_predict * cost_weights_partial_derivs_sums[layer])
                    biases[layer] += -step * (1.0/tot_classes_to_predict * cost_biases_partial_derivs_sums[layer])
                # 7) complete the average cost calculation
                # the cost for this iteration is
                # the sum of all the costs at the output layer
                # for each predicted class
                # over the total amount of predictable classes (i.e., avg)
                avg_cost = 1.0/tot_classes_to_predict * avg_cost
                # the avg cost for this iteration is a point in the curve
                # which we want to be as close to the min of that curve
                # as possible
                avg_cost_curve.append(avg_cost)
                iteration += 1
            return weights, biases, avg_cost_curve
        
        
    def predict_classes(self, weights, biases, inputs_x_curr):
        tot_classes_to_predict = self.nn_structure[2]
        classes = np.zeros((tot_classes_to_predict,))
        for i in range(tot_classes_to_predict):
            h, z = self.feed_forward(inputs_x_curr, weights, biases)
            print h[3]
            classes[i] = np.argmax(h[len(self.nn_structure)])
            print 'classes[i]'
            print classes[i]
        print 'predicted classes:'
        print classes
        return classes
        
        
    def test_nn(self, weights, biases, x_test_set):
        prediction = []
        for sample_index in range(0, len(x_test_set)):
            print('Testing NN with sample #{}').format(sample_index)
            sample_prediction = self.predict_classes(weights, biases, x_test_set[sample_index])
            single_output = self.convert_multi_to_single_output(sample_prediction)
            print 'single output'
            print single_output
            prediction.append(single_output)
        return prediction

        

    