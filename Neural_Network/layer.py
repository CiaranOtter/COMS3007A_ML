class Layer:
    def __init__(self, nodes):
        self.conns = [];
        self.Nodes = nodes;

    def addConn(self, conn):
        self.conns.append(conn)  

    def printLayer(self):
        print("Number of nodes: "+str(self.countNodes()))
        for n in self.Nodes:
            print("\t"+n.getNodeInfo())
        print("Number of connections: "+str(self.countConnections()))
        for c in self.conns:
            c.printConn()
        print("-----------------------------------------------------")

    def countConnections(self):
        return len(self.conns)

    def countNodes(self):
        return len(self.Nodes)

    def calcAllX(self):
        for n in self.Nodes:
            n.calculateNewX();

    def findSums(self):
        self.resetSums()
        self.calcAllX();
    
        for c in self.conns:
            c.incSum();

    def calcAllX(self):
        for n in self.Nodes:
            n.calculateNewX()

    def resetSums(self):
        for c in self.conns:
            c.resetChild()

    def getNodes(self):
        return self.Nodes

    def setIns(self, x):
        for i, n in enumerate(self.Nodes):
            n.setInputs(x[i])

    # def setEndError(self, y):
    #     for 
