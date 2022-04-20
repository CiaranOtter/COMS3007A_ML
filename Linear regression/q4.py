import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import logsumexp

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
    out = 0
    for i in range(len(theta)):
        out += theta[i]*(x**i)
    return out

for i, d in enumerate(design_matrix):
    y_values[i] = f(x_values[i],theta)+np.random.normal(loc=0, scale=8, size=1)

# ----- (v) -----

def PlotScatter(theta, theta_grad, theta_reg, x):
    plt.scatter(x_values, y_values)
    plt.plot(x[:,1], f(x[:,1],theta), "r")
    plt.plot(x[:,1], f(x[:,1], theta_grad), "g")
    plt.plot(x[:,1], f(x[:,1], theta_reg), "y")
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
    for d in training:
        temp = logsumexp((f(d[0],theta) - d[1] ))
        err += ((temp**2)/2)

    return err
        
print(Error(trained))

# ----- (iii) -----

x_train = design_matrix[design_matrix[:,1].argsort()]


# ----- (iv) -----

def Plot_Time_Error(time_step, training_error):
    plt.plot(time_step[:100:], training_error[:100:], "y")
    plt.show()

def Gradient_Decent(n):
    alpha = 0.0000001
    e = 10**(-5)

    theta_new = np.zeros(n)
    theta_old = np.random.uniform(0,1,n)

    print(theta_new)
    print(theta_old)

    time_step = np.array([]).astype(int)
    training_error = np.array([])
    index = 0

    while (np.linalg.norm(abs(theta_old-theta_new),2) >= e):
        for i in range(n):
            theta_old[i] = theta_new[i]
            theta_new[i] = theta_new[i] - Partial_Derivative(alpha,i, theta_old)
    
        time_step = np.append(time_step,index)
        index = index +1
        if (index%20 == 0):
            training_error = np.append(training_error,Error(theta_new))
    
        

    # print("time_step", time_step)
    # print(index)
    # print("training_error", training_error)
    # Plot_Time_Error(time_step,training_error)
    return theta_new
    

def Partial_Derivative(alpha, i, theta):
    err = 0;
    for j,d in enumerate(training):
        y_hat = f(d[0], theta)
        err += (y_hat - d[1])*(d[0]**i)

    return alpha*err

print(theta)
print(trained)
gradient = Gradient_Decent(3)
print(gradient)




# ----- part c -----
# ----- {i} -----

def Regularisation():

    design_matrix = np.ones((len(x_values), 4))
    for i,x in enumerate(x_values):
        design_matrix[i][1] = x
        design_matrix[i][2] = x*x
        design_matrix[i][3] = x*x*x
    
    def Partial_Derivative_regularised(alpha, i, theta, l):
        err = 0;
        for j,d in enumerate(training):
            y_hat = f(d[0], theta)
            err += alpha*(y_hat - d[1])*(d[0]**i)

        return err + alpha*l*theta[i]

    def Gradient_Decent_Regularised():
        alpha = 0.00000000001
        e = 10**(-5)
        l = 0.3

        theta_old = np.zeros(4)
        theta_new = np.random.uniform(0,1,4)
        while (np.linalg.norm(abs(theta_old-theta_new),2) >= e):
            for i in range(4):
                theta_old[i] = theta_new[i]
                pDer = Partial_Derivative_regularised(alpha, i, theta_old, l)
                print(pDer)
                theta_new[i] = theta_new[i] - pDer
    
            # time_step = np.append(time_step,index)
            # index = index +1
            # if (index%20 == 0):
            #     training_error = np.append(training_error,Error(theta_new))
    
        return theta_new;

    
    cubic_theta = Gradient_Decent_Regularised()
    PlotScatter(trained, gradient, cubic_theta, x_train)


Regularisation()
