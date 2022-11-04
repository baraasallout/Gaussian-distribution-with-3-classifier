import numpy as np
def gauss_density_value(x, d, mean, covariance):
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) *np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))

#A question 2.23 part 1 on our book

dimensions = 3
mean = np.array([[1], [2], [2]])  # Mean
covariance = np.array([[1., 0.,0.],  [0., 5.,2.],[0., 2.,5.]])
x = np.array([[0.5], [0], [1]])
print(gauss_density_value(x, dimensions, mean, covariance))