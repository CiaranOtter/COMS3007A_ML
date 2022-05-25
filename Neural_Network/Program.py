# make imports for this project
from network import Network
import numpy as np
import helpers
#-----------------------------

# # function to initialize a neural network layer
# # params:
# #   - thisLayer (integer, number of nodes in current layer)
# #   - thatLayer (integer, number of nodes in the following layer)
# #
# # returns:
# #   - a tuple which contains a randomized weight matrix and a bias vector

# def initLayer(thisLayer, thatLayer):
#     weight_matrix = np.random.rand(thisLayer, thatLayer);
#     bias_vector = np.random.rand(thatLayer);

#     return (weight_matrix, bias_vector)

# #-------------------------------------------------------------------------

# # function to randomly initialise a network
# # params:
# #   - nodes (list [] of integers, number of neurons in each layer)
# #
# # returns:
# #   - a list of tuples of the network weights and their bias vectors

# def init_random_network(nodes):
#     weights = []
#     for i in range(len(nodes)-1):
#         weights.append(initLayer(nodes[i], nodes[i+1]))

#     return weights

# # ------------------------------------------------------------------------

# def propogate(network_params, data):
    
#     for d in data:
#         bias = SumNode(network_params[1], 1);
#         node = SumNode(network_params[0], data[0]);   
        


    
# sigmoid = lambda z : 1/(1+math.e**(-z));

# def SumNode(weights, inputs):
#     out = 0;

#     for i in range(len(weights)):
#         out += weights[i]*inputs[i];
    
#     return out

# # load the data for this project from the helper program
# data = helpers.load_mnist();

# print("Data:")
# print(data[1])
# print("\n")
# dimensions = [784,200,100,10]
# network = init_random_network(dimensions)

# print(network)
# helpers.check_my_network(dimensions, network)





layer1 = np.array([[-8,7,3,2],[-4,9,5,1]]);
layer2 = np.array([[-1,6,2]])
weights = np.array([
    layer1, layer2  
])

network = Network(weights)
output = network.calculateOutput([1,1.5, -1, 3])
network.printNetwork()

print(output)