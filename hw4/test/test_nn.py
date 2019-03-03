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

# for a static set of starting weights, input, and output, test that the
# calculation of the errors in the first round of learning is as we expect



# ... what else do we test about an NN?
