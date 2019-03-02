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
        self.f = activation_function if activation_function != None else sigmoid
        # name for the activation function for sensible printing
        self.f_name = act_fn_name if act_fn_name != None else "sigmoid"
        #the bias value to use for each neuron in the layer. Use default 0 b/c:
        #https://datascience.stackexchange.com/questions/17987/how-should-the-bias-be-initialized-and-regularized
        self.b = layer_bias if layer_bias != None else 0
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


    def set_node_weights(self,node_i,weights_vector):
        """Sets weights of a single node in the layer to the given values"""
        if len(self.wts[0]) != len(weights_vector):
            raise ValueError("Dimension mismatch between self.wts and input")
        else:
            for j in range(len(weights_vector)):
                self.wts[i][j] = weights_vector[j]


    def feedforward_layer(self,input_values):
        """
        Calculates the vector of outputs from the whole layer for a given
        input vector

        Input: array-like of values, same length as the layer's number of cols
        Output: np.array of the activation values from each neuron, given the input
        """
        return



# use a lambda instead of this and 'global sigmoid' in create_layer
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))

# def create_layer(n_nodes,n_inputs,bias=0):
#     """
#     Initializes a Layer of the given dimensions with xavier-normalized
#     random weights and a sigmoid activation function.
#     - Alter someday to support non-sigmoidal nodes...
#     - There's also a world where this should be part of the layer intialization
#     """
#     #use xavier initialization because:
#     # https://www.quora.com/What-are-good-initial-weights-in-a-neural-network
#     # and https://stackoverflow.com/questions/48641192/xavier-and-he-normal-initialization-difference
#     arr = np.random.randn(n_nodes,n_inputs) * np.sqrt(1/n_inputs)
#     #use bias at zero by default because:
#     #https://datascience.stackexchange.com/questions/17987/how-should-the-bias-be-initialized-and-regularized
#     return Layer(arr,bias,lambda z: 1/(1+np.exp(-z)),"sigmoidal")



class NeuralNetwork:
    """
    Represents a full neural network with sigmoidal activation functions and
    a Euclidean squared distances loss function with regularization

    layer_sizes : array-like describing net structure, e.g. [8,3,8]
    inputs_x : array-like for the values of the initial inputs to the network
    true_results_y: array-like for the known answers (labels) to the given input
    """
    # TODO implement sparsity ^rho in cost function??
    def __init__(self,layer_sizes,inputs_x,true_results_y):
        layer_dims = [ (layer_sizes[i],layer_sizes[i+1]) for i
                                                in range(len(layer_sizes)-1) ]
        self.layers = np.array([ Layer(n,i) for n,i in layer_dims ])
        self.x = inputs_x
        self.y = true_results_y

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



    def feedforward():
        """implements feeding forward through the whole network"""
        return

    def backpropagate():
        """implements backpropagating error and updating weights through the whole network"""
        return
