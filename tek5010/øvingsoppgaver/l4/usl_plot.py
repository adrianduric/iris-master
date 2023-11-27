# Import libraries
import numpy as np # standard library for numerical operations in python
from scipy.optimize import least_squares # library for NLS
import matplotlib.pyplot as plt # standard library for plotting in python

# The USL function
def USL(parameter, N):
    R = (N) / (1 + parameter[0] * (N - 1) + parameter[1] * N * (N - 1))
    return R

# Define some noisy USL data
xValues = np.linspace(1, 200, 200)
yValuesR1 = np.array([1 * USL((0, 0), n) for n in xValues])
yValuesR2 = np.array([1 * USL((0.0001, 0), n) for n in xValues])
yValuesR3 = np.array([1 * USL((0.0007, 0.0003), n) for n in xValues])
yValuesR4 = np.array([0.25 * USL((-0.0335, 0.00032), n) for n in xValues])

# Plot the results
plt.plot(xValues, yValuesR1, label='C=1, alpha=beta=0')
plt.plot(xValues, yValuesR2, label='C=1, alpha=0.0001, beta=0')
plt.plot(xValues, yValuesR3, label='C=1, alpha=0.0007, beta=0.0003')
plt.plot(xValues, yValuesR4, label='C=0.25, alpha=-0.0335, beta=0.00032')
plt.xlabel('Processes [N]') # x-axis label
plt.ylabel('Performance [R]') # y-axis label
plt.title('The Universal Scalability Law') # plot title
plt.legend()
plt.show()
