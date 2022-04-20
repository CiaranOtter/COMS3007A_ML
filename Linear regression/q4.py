import numpy as np
import matplotlib.pyplot as plt
import math

# ----- part a -----
# ------ (i) ------

x_values = np.random.normal(loc=0, scale=10, size=150)

# ----- (ii) -----

design_matrix = np.ones((len(x_values), 3))
for i,x in enumerate(x_values):
    design_matrix[i][1] = x
    design_matrix[i][2] = x*x

# print(design_matrix)

# ----- (iii) -----

theta = np.zeros(3)
for i in range(3):
    theta[i] = np.random.uniform()  

# ----- (iv) -----

y_values = np.zeros(150)

def f(x, theta):
    return theta[0]+theta[1]*x+theta[2]*x*x

for i, d in enumerate(design_matrix):
    y_values[i] = f(x_values[i],theta)+np.random.normal(loc=0, scale=8, size=1)

# ----- (v) -----

def PlotScatter(theta, x):
    plt.scatter(x_values, y_values)
    plt.plot(x[:,1], f(x[:,1],theta), "r")
    plt.show()


# ----- (vi) -----

def setdiff2d_list(arr1, arr2):
    out = []
    for e in arr1:
        if (not e in arr2):
            out.append(e)
    return np.array(out)

data = np.ones((150,2));
data[:,0] = x_values;
data[:,1] = y_values;

training = data[np.random.choice(data.shape[0], 105, replace=False), :]
temp = setdiff2d_list(data, training)
validation = temp[np.random.choice(temp.shape[0], 30, replace=False), :]
testing = setdiff2d_list(temp, validation)


# ----- part b -----
# ----- (i) -----

Xt = design_matrix.transpose()

trained = np.dot(Xt, design_matrix)
trained = np.linalg.inv(trained)
trained = np.dot(trained, Xt)
trained = np.dot(trained, y_values)


# ----- (ii) -----

def Error(theta):
    # err = np.dot(y_values.transpose(), y_values)
    # err = err - np.dot(theta.transpose(),np.dot(design_matrix.transpose(), y_values))
    # err = err + np.dot(theta.transpose(), np.dot(design_matrix.transpose(),np.dot(design_matrix, theta)))
    # return err/2
    err = 0;
    for i in range(len(y_values)):
        temp = ( f(x_values[i],theta) - y_values[i])
        err += temp*temp

    return err/2
        
print(Error(trained))

# ----- (iii) -----

x_train = design_matrix[design_matrix[:,1].argsort()]

PlotScatter(trained, x_train)

# ----- (iv) -----

def Gradient_Decent():
    alpha = 0.01
    xi = 0
    theta_curr = np.random.rand(3)
    theta_old = np.zeros(3)

    while (np.linalg.norm(theta_curr-theta_old) > 0.001):
        theta_old = theta_curr
        y_pred = f(x_values, theta_old)
        print(y_pred)
        print(alpha*(-y_values)*design_matrix[:,0])
        theta_curr[0] = theta_old[0] - alpha*(f(x_values, theta_old)-y_values)*design_matrix[:,0]
        theta_curr[1] = theta_old[1] - alpha*(f(x_values, theta_old)-y_values)*design_matrix[:,1]
        theta_curr[2] = theta_old[2] - alpha*(f(x_values, theta_old)-y_values)*design_matrix[:,2]


        print(theta_curr)

Gradient_Decent()