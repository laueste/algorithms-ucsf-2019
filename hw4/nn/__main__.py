# implement command line stuff for the basic neural network tests
import numpy as np

from .utils import NeuralNetwork

test_net = NeuralNetwork([8,3,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8])
print(test_net)
print(test_net.layers[0].wts.shape)
