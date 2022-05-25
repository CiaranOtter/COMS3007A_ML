from node import Network_Node
from layer import Layer
from conn import Conn

class Network:
    def __init__(self, weights):
        self.Layers = []

        matrix = weights[0]
        shape = matrix.shape
        parents = []
        children = []

        for i in range(shape[1]):
            if (i == 0):
                parents.append(Network_Node(True))
                continue

            parents.append(Network_Node(False))

        self.inputNodes = parents
        for matrix in weights:
            shape = matrix.shape
            
            layer = layer(parents)
            # create a list of children
            for i in range(shape[0]):
                children.append(Network_Node(False)) 
            for p in range(shape[1]):
                for c in range(shape[0]):
                    conn = Conn(children[c], parents[p], matrix[c][p])
                    layer.addConn(conn)
            self.Layers.append(layer)
            bias_node = Network_Node(True)
            bias_node.setX(1)
            children.insert(0,bias_node)
            parents = children;
            children = []

        layer = layer(parents)
        self.Layers.append(layer)

    def printNetwork(self):
        for layer in self.Layers:
            layer.printLayer()

    def calculateOutput(self,inputs):
        Nodes = self.Layers[0].setIns(inputs)

        for i in range(len(self.Layers)):
            self.Layers[i].findSums()
        
        return self.getOutput()

    def getOutput(self):
        outNodes = self.Layers[-1].getNodes()
        out = []
        for n in outNodes:
            out.append(n.getX())

        return out

    def BackPropagation(self, x, y):
        layer = self.Layers[-1];
        layer.setEndError(y)
        

