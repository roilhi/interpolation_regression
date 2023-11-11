import numpy as np
import matplotlib.pyplot as plt

def eval_poly(b,x_data,x_axis_points):
    """
    Evaluates the n-th order Netwon polynomial given by:
    f_n(x) = b_0 + b_1(x-x_0) + b2(x-x_0)(x-x_1) + ... 
    where the inputs are:
    b: the set of Newton interpolation coefficients
    x_data: the set of x values to be interpolated (numpy array)
    x_axis_points: the set of x values to be evaluated by the polinomial 
    output: the set of f(x_axis_points) values of the built Newton polynomial
"""
    n = len(x_data)-1 #polynomial n-th order when having n+1 poiunts
    p = b[n]
    for i in range (1,n+1):
        p = b[n-i] + (x_axis_points-x_data[n-i])*p
    return p

def newton_coeffs(x_data,y_data):
    """
    From a set of points (x_data,y_data) calculates the bn interpolation coefficients
    for the Newton polynomial:
    f_n(x) = b_0 + b_1(x-x_0) + b2(x-x_0)(x-x_1) + ... 
    Inputs:
    x_data: the set of x values (numpy array)
    y_data: the set of y values (numpy array)
    """
    m = len(x_data)
    b = y_data.copy()
    for i in range(1,m):
        b[i:m] = (b[i:m]-b[i-1])/(x_data[i:m]-x_data[i-1])
    return b

# Example values:
x = np.array([1.6,2,2.5,3.2])
y = np.array([2,8,14,15])

# calculate newton coefficients
b = newton_coeffs(x,y)

# to build polynomial and evaluate
x_axis_values = np.linspace(0,5,10)

# newton polynomials
f1x = eval_poly(b[0:2], x[0:2], x_axis_values)
f2x = eval_poly(b[0:3], x[0:3], x_axis_values)
f3x = eval_poly(b, x, x_axis_values)

# plotting results
fig = plt.figure(figsize=(5,6))
plt.scatter(x,y)
plt.scatter(x,y)
plt.plot(x_axis_values,f1x, color='g')
plt.plot(x_axis_values,f2x, color='m')
plt.plot(x_axis_values,f3x, color='r')
plt.legend(['data','$f_1(x)$','$f_2(x)$','$f_3(x)$'])
plt.grid()
plt.show()