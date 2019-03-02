# implement command line stuff for the basic neural network tests
import numpy as np

from .utils import NeuralNetwork


test_net = NeuralNetwork([8,3,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8])

print("x:",test_net.x)
print("y:",test_net.y)

for i,layer in enumerate(test_net.layers):
    print(i)
    print(layer)
    print()
