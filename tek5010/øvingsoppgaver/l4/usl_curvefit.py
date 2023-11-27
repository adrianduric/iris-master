# Import libraries
import numpy as np # standard library for numerical operations in python
from scipy.optimize import least_squares # library for NLS
import matplotlib.pyplot as plt # standard library for plotting in python

# The USL function
def USL(parameter, N):
    R = (N) / (1 + parameter[0] * (N - 1) + parameter[1] * N * (N - 1))
    return R

# NLS
def fun(parameter):
    return USL(parameter, xValues) - yUSLdata

# Define some noisy USL data
xValues = np.array([0, 15, 30, 60, 100, 125, 150]) # the x-axis values
yValuesR1 = np.array([0, 7, 24, 22, 25, 29, 26]) # the R1 data set
yValuesR2 = np.array([0, 20, 50, 64, 30, 26, 15]) # the R2 data set
#yUSLdata = yValuesR1 # set data set to regress on
yUSLdata = yValuesR2 # set data set to regress on


# Estimate the USL parameters using scipy NLS
guessParameterUSL = [0.0, 0.000] # initial parameter guess
result = least_squares(fun, guessParameterUSL) # call the NLS in scipy

# Get the USL parameters from the result of the regression
alphaEstimated = result.x[0] # the estimated alpha parameter
betaEstimated = result.x[1] # the estimated beta parameter
yUSLEstimated = USL([alphaEstimated, betaEstimated], xValues) # graph USL using estimated parameters

# Plot the results
plt.plot(xValues, yUSLdata, '*') # plot the noisy USL data
plt.plot(xValues, yUSLEstimated) # plot USL using estimated parameters
plt.xlabel('Processes [N]') # x-axis label
plt.ylabel('Performance [R]') # y-axis label
plt.title('The Universal Scalability Law') # plot title
LegendTextUSLEstimated = 'alpha=' + "%.4f" %alphaEstimated + ', beta=' + "%.4f" %betaEstimated
plt.legend([[], LegendTextUSLEstimated]) # legendes
print(result) # print out the numerical results of the regression to screen
plt.show()
