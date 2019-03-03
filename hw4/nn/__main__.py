# implement command line stuff for the basic neural network tests
import numpy as np

from .utils import NeuralNetwork

# test_net = NeuralNetwork([8,3,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8])
# print(test_net)
# print()
sig = lambda z: 1/(1+np.exp(-z))
mini_net = NeuralNetwork([2,2,1],[1,0],[2])
input = np.array([1,0])
weights_1 = np.array([[0.4,0.6],[0.9,0.1]])
weights_2 = np.array([[0.45,0.55]])
mini_net.layers[0].set_weights_arr(weights_1)
mini_net.layers[1].set_weights_arr(weights_2)
ff1 = sig(np.matmul(weights_1,input))
ff1_real = mini_net.layers[0].feedforward_layer(input)
ff2 = sig(np.matmul(weights_2,ff1))
ff2_real = mini_net.layers[1].feedforward_layer(ff1_real)
full_ff_real = mini_net.feedforward()


print("Expected Feedforward 1:",np.matmul(weights_1,np.array([1,0])),ff1)
print("Actual Feedforward 1:",ff1_real)
print("Expected Feedforward 2:",np.matmul(weights_2,ff1),ff2)
print("Actual Feedforward 2:",ff2_real)
print("NeuralNetwork level FF:",full_ff_real)
