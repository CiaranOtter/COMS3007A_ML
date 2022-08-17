import math
import random
import numpy as np
from sklearn.model_selection import train_test_split
import time

print("------------------------------")
print("welcome to th machine learner")
print("------------------------------")
print("\n")

sigmoid = lambda z : 1/(1+np.exp(z))
der = lambda a : a*(1-a)

def init(inputSize, outputSize, noHiddenLayers, hiddenweightSize):
    if (noHiddenLayers == None):
        noHiddenLayers = random.randint(1,10);

    HiddenLayerSizes = [ int(hiddenweightSize/(i+1)) for i in range(noHiddenLayers) ];
    weights = []

    inputWeight = np.random.rand(HiddenLayerSizes[0], inputSize+1);
    weights.append(inputWeight);

    for i in range(noHiddenLayers-1):
        tempWeight = np.random.rand(HiddenLayerSizes[i+1], HiddenLayerSizes[i]+1)
        weights.append(tempWeight)

    weights.append(np.random.rand(outputSize, HiddenLayerSizes[-1]+1))

    # displaying the data regarding the size and weights of the netork

    print("--------------------------------------------------")
    print("input size: " + str(inputSize))
    print("output size: "+str(outputSize))
    print("number of Hidden Layers: "+str(noHiddenLayers))
    print("Hidden layer sizes: "+str(HiddenLayerSizes))
    # print("Weights: "+str(weights))
    print("---------------------------------------------------")
    print("\n")

    return weights

def fetchData_Labeled(inputFile, labelFile):
    inputs = np.loadtxt('data/inputs.txt');
    labels = np.loadtxt('data/labels.txt')
    return inputs, labels

def train(weights, inputValues, ExpectedValues, i):
    delta = []
    for i in range(len(weights)):
        delta.append(np.zeros((np.shape(weights[i]))))
    delta = np.array(delta)
    predictedValues, a = forward(inputValues, weights)
    newWeights, errors = backward(predictedValues, ExpectedValues, weights, a)
    newWeights = updateWeights(newWeights, errors, a, delta, i)
    return newWeights, errors

def forward(inputValues, weights):

    a = []

    for i in range(len(weights)):
        inputValues = np.insert(inputValues, 0, [1], axis=0)
        a.append(inputValues)
        z = np.dot(weights[i], inputValues)
        inputValues = sigmoid(z)

    a.append(inputValues)
    return inputValues, a

    

def backward(predicted, expected, weights, a):
    errors = []

    error = (predicted - expected)
    errors.append(error)
    error = np.insert(error, 0, [1], axis=0)

    for i in range(len(weights)-1,0, -1):
        error = np.delete(error, 0, 0)
        error =np.dot(np.transpose(weights[i]), error)
        error = error * der(a[i])

        errors.append(error)

    return weights, errors


def updateWeights(weights, errors, a, delta, n):
    alpha = 0.5
    reg = 0.8
    errors = errors[::-1]
        
    for i in range(len(errors)):
        if (i != len(errors)-1):
            errors[i] = np.delete(errors[i], 0,0)
        deltat = np.dot(errors[i], np.transpose(a[i]))
        delta[i] = ((1/n)*delta[i])+reg*deltat

        weights[i] = weights[i] + alpha*delta[i]
        

    return weights

inputs, labels = fetchData_Labeled("data/inputs.txt","data/labels.txt")

data_train, data_test, labels_train, labels_test = train_test_split(inputs, labels, test_size=0.40)
data_test, data_val, labels_test, labels_val = train_test_split(data_test, labels_test, test_size=0.5)




def checkError(array):
    for i in range(len(array)):
        array[i] = np.reshape(array[i], -1)
        for j in range(len(array[i])):
            if abs(array[i][j]) >= 0.001:
                return False;
    return True


# initialising the weights
Weights = init(np.shape(inputs)[1], 10, 5, np.shape(inputs)[1])
initWeights = Weights

# looping through tarining data to train the model
start = time.time()
for i in range(int(len(data_train))):
    end = time.time();
    timeDiff = end - start
    start = end
    print("time taken for iteration: ", timeDiff)
    print("\n===================================== Iteration "+str(i)+" ======================================================\n")
    expected = np.zeros(10)
    expected[int(labels_train[i])] = 1
    count = 1
    while True:
        Weights, errors = train(Weights, np.transpose(np.array([data_train[i]])), np.transpose(np.array([expected])), count)
        count  += 1
        errors = np.array(errors)
        if (checkError(np.array(errors))):
            break

noAccurate =0;
noWrong = 0;
for i in range(len(data_test)):
    print("expected output", labels_test[i])
    print("test input Values: ", np.transpose(np.array([data_test[i]])))
    predicted, a = forward(np.transpose(np.array([data_test[i]])), Weights)
    print("predicted value is: ", predicted)
    maxt = 0
    index = 0

    for j in range(len(predicted)-1):
        if (predicted[j][0] > maxt):
            maxt = predicted[j][0]
            index = j


    if (index == int(labels_test[i])):
        noAccurate += 1
    else:
        noWrong += 1

print("total predictions correct: ", noAccurate)
print("total wrong: ", noWrong)
