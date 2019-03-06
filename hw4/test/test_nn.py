# test functions for the neural network class
from nn import utils
import numpy
import pytest

# test overall structure (aka that weight arrays are the right dimensions)
@pytest.mark.parametrize("structure,input,output,layer_sizes", [([5,3,7,10,1],
[0,0,0,0,0],[0],[(3,5),(7,3),(10,7),(1,10)])])
#the output and input values are irrelevant here, just the shapes

def test_structure(structure, input, output, layer_sizes):
    nn = utils.NeuralNetwork(structure,input,output)
    for i,layer in enumerate(nn.layers):
        assert layer.wts.shape == layer_sizes[i]


# test layer feedforward output (1D array of len(N_nodes))
@pytest.mark.parametrize("n_nodes,n_inputs,x,weights,raw_answer",
[(2,2,numpy.array([10,20]),numpy.array([[1,1],[0,2]]),numpy.array([30,40]))])

def test_layer_feedforward(n_nodes, n_inputs, x, weights, raw_answer):
    l = utils.Layer(n_nodes,n_inputs)
    l.set_weights_arr(weights)
    ff_output = l.feedforward_layer(x)
    assert (ff_output == l.f(raw_answer)).all()


## TODO: convert to pytest form:

def test_feedforward_results():
    sig = lambda z: 1/(1+np.exp(-z))
    mini_net = NeuralNetwork([2,2,1],[1,0],[2])
    input = np.array([1,0])
    weights_1 = np.array([[0.4,0.6],[0.9,0.1]])
    weights_2 = np.array([[0.45,0.55]])
    mini_net.layers[0].b = 1 #change bias from zero so we can test it
    mini_net.layers[0].set_weights_arr(weights_1)
    mini_net.layers[1].set_weights_arr(weights_2)
    ff1 = sig(np.matmul(weights_1,input) + 1)
    ff1_real = mini_net.layers[0].feedforward_layer(input)
    ff2 = sig(np.matmul(weights_2,ff1))
    ff2_real = mini_net.layers[1].feedforward_layer(ff1_real)
    full_ff_real = mini_net.feedforward()


    print("Expected Feedforward 1:",sig(np.array([1*0.4+0*0.6, 1*0.9+0*0.1])+1),ff1)
    print("Actual Feedforward 1:",ff1_real)
    print("Expected Feedforward 2:",np.matmul(weights_2,ff1),ff2)
    print("Actual Feedforward 2:",ff2_real)
    print("NeuralNetwork level FF:",full_ff_real)



# ... what else do we test about an NN?
