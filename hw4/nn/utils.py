# classes and general utility functions for implementing a neural network
import numpy as np

class Layer:
    """
    Represents a single layer of a neural network.
    Layer is a 2D numpy array of the given dimensions with xavier-normalized
    random weights and a sigmoid activation function.

    - Alter someday to support non-sigmoidal nodes...

    Example array:
            Input1, Input2, Input3
    Node1     0.3    0.15    2.7
    Node2     0.8     1.6    5.6
    Node3     4.1     0.9    0.1
    Node4     3.2     2.4    0.03
    """

    def __init__(self, n_nodes, n_inputs,
                    layer_bias=None,
                    activation_function=None,
                    act_fn_name=None):
        #the activation function for each neuron in the layer
        sigmoid = lambda z: 1 / (1+np.exp(-z))
        self.f = activation_function if activation_function is not None else sigmoid
        # name for the activation function for sensible printing
        self.f_name = act_fn_name if act_fn_name is not None else "sigmoid"
        #the bias value to use for each neuron in the layer. Use default 0 b/c:
        #https://datascience.stackexchange.com/questions/17987/how-should-the-bias-be-initialized-and-regularized
        self.b = layer_bias if layer_bias is not None else 0
        #2D np array of weights, nodes(r) x inputs(c)
        #use xavier initialization because:
        # https://www.quora.com/What-are-good-initial-weights-in-a-neural-network
        # and https://stackoverflow.com/questions/48641192/xavier-and-he-normal-initialization-difference
        arr = np.random.randn(n_inputs,n_nodes) * np.sqrt(1/n_inputs)
        self.wts = arr

    # Overload the __repr__ operator to make printing simpler.
    def __repr__(self):
        #make list so "array()" not printed!
        lines = []
        for node in self.wts:
            lines.append( str([ round(e,5) for e in node ]) )
        lines.append("Bias: %s" % self.b)
        lines.append("Activation Function: %s" % self.f_name)
        return "\n".join(lines)

    def get_node(self, i):
        if i >= len(self.wts):
            raise IndexError("Node index %s out of range for %s nodes" % (i,len(self.wts)))
        else:
            return self.wts[i]

    def get_dim(self):
        return self.wts.shape

    def set_bias(self,new_bias):
        """Replaces the layer's bias term with the input value"""
        self.b=new_bias

    def set_weights_arr(self,array,change_dimensions_okay=False):
        """Replaces the weights array with the given array. Does not allow
        the new array to be a different shape unless the optional arg is True"""
        if array.shape == self.wts.shape or change_dimensions_okay == True:
            self.wts = array
        else:
            raise ValueError("Dimensions Mismatch with self.wts disallowed: \
                    %s self vs %s input" % (self.wts.shape,array.shape))


    def set_node_weights(self,node_i,weights_vector):
        """Sets weights of a single node in the layer to the given values"""
        if len(self.wts[0]) != len(weights_vector):
            raise ValueError("Dimension Mismatch between self.wts and input")
        else:
            for j in range(len(weights_vector)):
                self.wts[i][j] = weights_vector[j]


    def feedforward_layer(self,input_values):
        """
        Calculates the vector of outputs from the whole layer for a given
        input vector:

        activation_function(W0x0 + W1x1 + W2x2 + ...)  for each node in layer

        Input: array-like of values, same length as the layer's number of cols
        Output: np.array of the activation values from each neuron, given the input
        """
        neuron_outputs = np.matmul(self.wts,input_values) + self.b #term aka 'z'
        return self.f(neuron_outputs) #apply AF to z






def single_example_loss(y,true_y):
    """Returns half the square of the Euclidean distance between y and true_y"""
    return ( np.abs(true_y-y)^2 ) * 0.5

def loss(nn,ys,true_ys,wd_param=0.001):
    """Returns single example loss plus weight decay term applied across
    an array of many output answers vs correct answers"""
    #source for some insight into reasonable hyperparameter values
    #https://yashuseth.blog/2018/11/26/hyper-parameter-tuning-best-practices-learning-rate-batch-size-momentum-weight-decay/
    # TODO ideally test 0, 0.001, 0.0001, and 0.00001 as weight decay params?
    # other sources mention lambda up to 5.0 though?? talk to Jared and Taylor
    if len(ys) != len(true_ys):
        raise ValueError("Dimension Mismatch between input arrays of outputs and labels: %s vs %s" %(len(ys),len(true_ys)) )
    avg_sum_squares_err = np.mean(single_example_loss(ys,true_ys))
    n_weights = np.sum(np.prod([l.wts.shape for l in nn.layers],axis=1))
    sum_weight_values = np.sum([np.sum(l.wts) for l in nn.layers])
    weight_decay_term = wd_param * (sum_weight_values / n_weights)
    return avg_sum_squares_err + weight_decay_term





class NeuralNetwork:
    """
    Represents a full neural network with sigmoidal activation functions an
    a Euclidean squared distances loss function with regularization, starting
    biases at 0 and xavier normalization applied to the starting random weights.

    layer_sizes : array-like describing network structure, e.g. [8,3,8],
    inputs_x : array-like for the values of the initial inputs to the network
    true_results_y: array-like for the known answers (labels) to the given input
    alpha: the step size
    wd: the weight decay parameter
    """
    # TODO implement sparsity ^rho in cost function??
    def __init__(self,layer_sizes,inputs_x,true_results_y,alpha=3,wd=0.0001):
        # make layers for every transition, including inputs->first layer
        layer_dims = [(len(inputs_x),layer_sizes[0])] + [
          (layer_sizes[i],layer_sizes[i+1]) for i in range(len(layer_sizes)-1) ]
        self.layers = np.array([ Layer(n,i) for n,i in layer_dims ])
        self.x = inputs_x
        self.y = true_results_y
        self.alpha = alpha
        self.wd = wd
        self.m = len(self.layers)

    def __repr__(self):
        #much easier to read printing version
        lines = []
        lines.append("x: %s" % self.x)
        lines.append("y: %s" % self.y)
        lines.append("Layers:")
        for i,layer in enumerate(self.layers):
            lines.append("%s -" %i)
            lines.append(layer.__repr__())
        return "\n".join(lines)

    def get_blank_weights_array(self):
        return np.array([ np.zeros(l.wts.shape) for l in self.layers ])

    def feedforward(self,new_input=None,return_activations=False):
        """
        implements feeding forward through the whole network

        Input: optional array of inputs, optional flag to return all activations
               (if no inputs given, use self.x)
        Output: Array of output values from the final layer
                        -OR- (if return_activations set to True)
                Tuple of (array of outputs, array of activations for each layer)
        """
        #allow but don't require passing in new inputs, copy so we don't alter
        input = np.copy(new_input) if new_input is not None else np.copy(self.x)
        if return_activations == True:
            activations = np.array([np.zeros(l.wts.shape) for l in self.layers])
            for i,layer in enumerate(self.layers):
                output = layer.feedforward_layer(input)
                activations[i] = output
                input = output
            return output,activations
        else:
            for layer in self.layers:
                input = layer.feedforward_layer(input)
            return input

    def backpropagate(self,new_y=None,new_x=None):
        """
        implements backpropagating error and updating weights through the whole
        network

        Inputs: optional array of true outputs and optional array of inputs
                (if not given, uses self.y and self.x respectively)
        Outputs: None (updates weights and biases in self.layers)
        """
        print("Beginning Backpropagation")
        #allow but don't require passing in true output labels, copy for safety
        true_y = np.copy(new_y) if new_y is not None else np.copy(self.y)
        input = np.copy(new_x) if new_x is not None else np.copy(self.x)
        output,activations = self.feedforward(input,return_activations=True)

        # initialize arrays to hold updated weights and biases
        new_weights = np.array([ np.zeros(l.wts.shape) for l in self.layers ])
        new_biases = np.zeros(self.m)

        # initialize array to hold deltas for each layer, computer outer delta
        deltas = np.array([ np.zeros(l.wts.shape) for l in self.layers ])

        # Fill output layer values
        # NOTE: Hadamard product is numpy default vector * operation
        # NOTE: for sigmoid, derivative s'(z) = s(z) * (1-s(z))
        delta_outer = -(true_y-output) * ( output*(1-output) )
        new_weights[-1] = delta_outer*np.transpose(activations[-1])
        new_biases[-1] = delta_outer
        deltas[-1] = delta_outer

        # Iterate through all non-output layers in reverse, with their activations
        for i in range(self.m-2,-1,-1):
            l = self.layers[i]
            a = activations[i]
            ## TODO: There must be a better indexing strategy than this...
            # calculate delta from the delta of the next layer
            delta = ( np.transpose(l.wts)*deltas[i+1] ) * ( a*(1-a) )
            deltas[i] = delta
            # calculate layer's errors from the delta of the next layer
            new_weights[i] = deltas[i+1]*np.transpose(a)
            new_biases[i] = deltas[i+1]

        # Iterate through all layers forward, updating weights
        for i,layer in enumerate(self.layers):
            #print("   Layer",i,":")
            weight_decay_term = self.wd*layer.wts
            error_term = (1/self.m) * new_weights[i]
            updated_wts = layer.wts - self.alpha*(error_term+weight_decay_term)
            updated_bias = layer.b - self.alpha*(1/self.m)*new_biases[i]
            #print("Delta:",deltas[i])
            #print("Partial Deriv:",new_weights[i])
            #print("Updated Weights:",updated_wts)
            layer.set_weights_arr(updated_wts)
            layer.set_bias(updated_bias)

        #print("Deltas",deltas)
        #print("Partial Derivs",new_weights)
        #print("Partial Bias Derivs",new_biases)
        #print("New Weights",[ [np.round(r,3) for r in l.wts] for l in self.layers ])
