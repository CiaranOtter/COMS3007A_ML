import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import logsumexp
# ----- QUESTION 4 -----
# ----- part a -----
# ------ (i) ------

# Sample 150 x-values for the model

x_values = np.random.normal(loc=0, scale=10, size=150)

# ----- (ii) -----

    # compute the design matrix for the model

design_matrix = np.ones((len(x_values), 3))
for i,x in enumerate(x_values):
    design_matrix[i][1] = x
    design_matrix[i][2] = x*x

# ----- (iii) -----

# sample three true value for Theta 

theta = np.zeros(3)
for i in range(3):
    theta[i] = np.random.uniform()  

# ----- (iv) -----

# use the design matrix and theta to calculate true y_values

y_values = np.zeros(150)

def f(x, theta):
    out = 0
    for i in range(len(theta)):
        out += theta[i]*(x**i)
    return out

for i, d in enumerate(design_matrix):
    y_values[i] = f(x_values[i],theta)+np.random.normal(loc=0, scale=8, size=1) # adding some noise to each data point

# ----- (v) -----

# plot a scatter plot of teh x and y values

def PlotScatter():
    plt.scatter(x_values, y_values)
    plt.title("Scatter plot of generated x and y values")

PlotScatter()
plt.show()

# ----- (vi) -----

# split the data into traing testing and validation data

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

# use the Moore-penrose pseudo-inverse to calculate the values of theat

Xt = design_matrix.transpose()

trained = np.dot(Xt, design_matrix)
trained = np.linalg.inv(trained)
trained = np.dot(trained, Xt)
trained = np.dot(trained, y_values)

print("Moore-penrose (closed form solution):")
print("------------------------------------\n")

# ----- (ii) -----

# calculate difference between true values and calculated values
print("True value of Theta: ", theta)
print("Moore-Penrose solution: ", trained)
print("difference between True and predicted values: ", theta - trained)
print("\n")


# ----- (iii) -----

# calculate training and validation error for learned model

def Error(theta, data):
    err = 0;
    for d in data:
        temp = ( d[1] - f(d[0],theta))
        err += (temp**2)

    return err/2
        
print("Training error for Moore-Penrose: ", Error(trained, training))
print("Validation error for Moore-Penrose: ", Error(trained, validation))
print("\n")

# ----- (iv) -----

# plot scatter polot for x and y values and the Moore-Penrose prediction in red

PlotScatter()
x_train = design_matrix[design_matrix[:,1].argsort()]
plt.plot(x_train[:,1], f(x_train[:,1], trained), "r")
plt.title("Plot of scatter points and Moore-Penrose Prediction for theta (red)")
plt.show()

# ----- (v) -----
# repeat previous stesp but for Gradient descent 

# function for calculating gradient descent

print("Gradient descent solution:")
print("-------------------------\n")
def Gradient_Decent(n, alpha, e, l):

    theta_new = np.zeros(n)
    theta_old = np.random.uniform(0,1,n)

    time_step = np.array([]).astype(int)
    training_error = np.array([])
    index = 0

    while (np.linalg.norm(abs(theta_old-theta_new),2) >= e):
        for i in range(n):
            theta_old[i] = theta_new[i]
            theta_new[i] = theta_new[i] - Partial_Derivative(alpha,i, theta_old, 0)
    
        time_step = np.append(time_step,index)
        index = index +1
        if (index%20 == 0):
            training_error = np.append(training_error,Error(theta_new, training))
    
    # plot the training error over time

    

    return theta_new, time_step, training_error    

def Partial_Derivative(alpha, i, theta, l):
    err = 0;
    for j,d in enumerate(training):
        y_hat = f(d[0], theta)
        err += alpha*(y_hat - d[1])*(d[0]**i)+alpha*l*theta[i]

    return err

gradient, ts, te = Gradient_Decent(3, 0.0000001, 10**(-5), 0)

# ----- [4b] (ii) {Gradient descent} ----- 
# calculating difference between true values of theat and theta predicted by gradient descent

print("True values of Theat: ", theta)
print("Gradient descent Theta: ", gradient)
print("difference between True and predicted values: ", theta- gradient)
print("\n")
# ----- [4b] (iii) {Gradient} ----- 
# computing errrors in validation and training for gradient descent

print("Training error for gradient descent: ", Error(gradient, training))
print("Validation error for gradient descent: ", Error(gradient, validation))
print("\n")

# ----- [4b] (iv) {Gradient descent} -----
# plotting the model

PlotScatter()
plt.plot(x_train[:,1], f(x_train[:,1], gradient), "r")
plt.title("Plot of scatter points and Gradient descent Prediction for theta (red)")
plt.show()

# ----- [4b] (v) {Gradient descent} -----
# plot error time 

plt.plot(ts[:100:], te[:100:], "y")
plt.title("plot for training error for regression model over time ")
plt.show()

# ----- part c -----
# ----- (i) -----
# appending a third feature to design matrix 

print("Third Order gradient descent (no regularization): ")
print("------------------------------------------------\n")

third_feature = np.zeros(150)
for i,x in enumerate(x_values):
    third_feature[i] = x*x*x

design_matrix = np.c_[design_matrix, third_feature]

# ----- (ii) -----

# -----[4b] (ii) {third order gradient descent} -----
# calculating difference between true values of theat and theta predicted by gradient descent

gradient_third_order, ts, te = Gradient_Decent(4,0.000000001, 10**(-6), 0);

theta_third_order = np.append(theta, 0)
print("True values of Theat: ", theta_third_order)
print("Gradient descent Theta: ", gradient_third_order)
print("difference between True and predicted values: ", theta_third_order - gradient_third_order)
print("\n")

# ----- [4b] (iii) {third order gradient descent} -----
# computing errrors in validation and training for gradient descent

print("Training error for gradient descent: ", Error(gradient_third_order, training))
print("Validation error for gradient descent: ", Error(gradient_third_order, validation))
print("\n")

# ----- [4b] (iv) {third order gradient descent} -----
# plotting the model

PlotScatter()
plt.plot(x_train[:,1], f(x_train[:,1], gradient_third_order) ,"r")
plt.title("Third order gradient descent solution (Overfitted)")
plt.show()

# ----- [4b] (v) {Third order gradient descent} -----
# plot error time 

plt.plot(ts[:100:], te[:100:], "y")
plt.title("plot for training error for regression model over time ")
plt.show()

# ----- (iii) -----
# Training using third orer gradient descent with regularization

# lambda set to 0.3 

gradient_reg, ts, te = Gradient_Decent(4, 0.000000001, 10**(-6), 0.3)
print("Third Order gradient descent with regularization: ")
print("------------------------------------------------\n")

# -----[4b] (ii) {third order gradient descent with regularization} -----
# calculating difference between true values of theat and theta predicted by gradient descent

print("True values of Theat: ", theta_third_order)
print("Gradient descent Theta: ", gradient_reg)
print("difference between True and predicted values: ", theta_third_order - gradient_reg)
print("\n")

# ----- [4b] (iii) {third order gradient descent with regularization} -----
# computing errrors in validation and training for gradient descent

print("Training error for third order gradient descent (regularization): ", Error(gradient_reg, training))
print("Validation error for gradient descent (regularization): ", Error(gradient_reg, validation))
print("\n")

# ----- [4b] (iv) {third order gradient descent with regularization} -----
# plotting the model

PlotScatter()
plt.plot(x_train[:,1], f(x_train[:,1], gradient_reg) ,"r")
plt.title("Third order gradient descent solution with regularization")
plt.show()

# ----- [4b] (v) {Third order gradient descent with regularization} -----
# plot error time 

plt.plot(ts[:100:], te[:100:], "y")
plt.title("plot for training error for regression model over time ")
plt.show()

