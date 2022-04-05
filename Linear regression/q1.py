import numpy as np

x = np.array([1,2,3,4,5])
y = np.array([1,3,2,3,5])


def f(x, theta):
    return theta[0] + theta[1]*x;


def Error(theta, x):
    temp = 0;
    for i in range(len(x)):
        temp += (f(x[i], theta) - y[x[i]-1])*(f(x[i], theta) - y[x[i]-1])
    
    return temp/2

X = np.zeros((len(x), 2));

X[:,0] = [1,1,1,1,1]
X[:,1] = x

i =0

tX = X.transpose();

theta = np.dot(tX, X)
theta = np.linalg.inv(theta)

theta = np.dot(theta,tX)
theta = np.dot(theta,y)

print(theta)
print(Error(theta, x))

def calcInput(x):
    return f(x, theta)

tin = int(input("estimate for y at x = "))
estimate = calcInput(tin)

print(Error(theta, [estimate]))

# only submitting question 4