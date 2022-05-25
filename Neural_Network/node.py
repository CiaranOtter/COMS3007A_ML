import math

class Network_Node:
    def __init__(self, bias):
        self.Sum = 0
        self.x = 0
        self.error = 0
        self.activation = lambda z: 1/(1+math.e**(-z))
        # self.gBack = lambda z: 
        self.bias = bias

    def setInputs(self, i):
        self.x = i

    def getSum(self):
        return self.Sum;

    def setSum(self, i):
        self.Sum = i;
    
    def incSum(self, i):
        self.Sum += i
    
    def getX(self):
        return self.x

    def setX(self, i):
        self.x = i

    def calculateNewX(self):
        if (self.getSum() == 0):
            return
        out = self.activation(self.getSum())
        self.setX(out)

    def getNodeInfo(self):
        string = "{ "
        string += "Sum: "+str(self.Sum)+", x: "+str(self.x)+", error: "+str(self.error)+" }"
        return string
