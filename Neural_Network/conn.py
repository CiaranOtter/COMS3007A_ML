class Conn:
    def __init__(self, child, Parent, weight):
        self.child = child
        self.Parent = Parent
        self.weight = weight
        self.inputNodes = []

    def printConn(self):
        print("\t"+self.Parent.getNodeInfo()+" - "+str(self.weight)+" -> "+self.child.getNodeInfo())

    def resetChild(self):
        self.child.setSum(0)

    def incSum(self):
        self.child.incSum(self.Parent.getX()*self.weight)