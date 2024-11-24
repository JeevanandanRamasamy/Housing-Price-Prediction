import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pylab
from ipywidgets import interact
from sklearn.linear_model import LinearRegression
import contextlib
from io import StringIO
from sklearn import metrics

np.random.seed(42)
plt.style.use('fivethirtyeight')
sns.set_context("talk")

# Load the data
df = pd.read_csv('USA_Housing.csv')
df_adjusted = df[['Avg. Area House Age', 'Avg. Area Number of Rooms', 'Price']] # keep only the columns "Avg. Area House Age" and "Avg. Area Number of Rooms" and Price
print(df_adjusted)

# Split the data into training and testing sets
df_training, df_testing = train_test_split(df_adjusted, test_size=0.1)

# Normalize the data
scaler = MinMaxScaler()
X_scaled_values = scaler.fit_transform(df_training[['Avg. Area House Age', 'Avg. Area Number of Rooms']])
Y_scaled_values = scaler.fit_transform(df_training[['Price']]).reshape(-1)
x = X_scaled_values
y = Y_scaled_values

# Plot Avg. Area House Age vs. Price
x_min, x_max = x[:,0].min(), x[:,0].max()
y_min, y_max = y.min(), y.max()
plt.scatter(x[:,0], y)
plt.plot([x_min, x_max], [y_min, y_max], color='red')
plt.xlabel('Avg. Area House Age')
plt.ylabel('Price')
plt.title('Scatter plot of Avg. Area House Age vs. Price')
plt.show()

# Plot Avg. Area Number of Rooms vs. Price
x_min, x_max = x[:,1].min(), x[:,1].max()
y_min, y_max = y.min(), y.max()
plt.scatter(x[:,1], y)
plt.plot([x_min, x_max], [y_min, y_max], color='red')
plt.xlabel('Avg. Area Number of Rooms')
plt.ylabel('Price')
plt.title('Scatter plot of Avg. Area Number of Rooms vs. Price')
plt.show()

def h(theta0, theta1, x):
    """Return the model theta0 + theta1*x"""
    return theta0 + theta1*x

def sqerror(x, y, theta0, theta1):
    """
    Input: parameters theta0 and theta1 of the model
           x, y vectors
    Returns: L2 square error
    Assumptions: none
    """
    y_pred = h(theta0, theta1, x)
    squared_error = (y_pred - y)**2
    return squared_error.mean()
print(sqerror(x[:,[0]], y, 0.29, 0.52))
print(sqerror(x[:,[1]], y, 0.29, 0.52))

def abserror(x, y, theta0, theta1):
    """
    Input: parameters theta0 and theta1 of the model 
           x, y vectors
    Returns: L1 error
    Assumptions: none
    """
    y_pred = h(theta0, theta1, x)
    absolute_error = abs(y_pred - y)
    return absolute_error.mean()
print(abserror(x[:,[0]], y, 0.29, 0.52))
print(abserror(x[:,[1]], y, 0.29, 0.52))

def huberror(x, y, theta0, theta1, delta):
    """
    Input: parameters theta0, theta1 and delta of the model 
           x, y vectors
    Returns: psuedo huber error
    Assumptions: none
    """
    y_pred = h(theta0, theta1, x)
    absolute_error = abs(y_pred - y)
    pseudo_huber_error = delta**2 * (np.sqrt(1 + (absolute_error/delta)**2) - 1)
    return pseudo_huber_error.mean()
print(huberror(x[:,[0]], y, 0.29, 0.52, 0.1))
print(huberror(x[:,[1]], y, 0.29, 0.52, 0.1))

def f(theta0, theta1, dim):
    """
    Plot the line and points in an interactive panel
    """
    # plot the line for theta0 and theta1
    y_pred = h(theta0, theta1, x[:,dim])
    # compose plot
    pylab.plot(x[:,dim], y_pred)
    
    # compute the L2 error for theta0 and theta1 for 5 decimal places
    sqerr = round(sqerror(x[:,[dim]], y, theta0, theta1), 5)
    # compute the absolute or L1 error for theta0 and theta1
    abserr = round(abserror(x[:,[dim]], y, theta0, theta1), 5)
    # compute the phub error for theta0 and theta1
    huberr = round(huberror(x[:,[dim]], y, theta0, theta1, 0.01), 5)
    pylab.title('L1=' + str(abserr) + '  L2=' + str(sqerr) + '  hub=' + str(huberr))
    
    # plot the points
    x1 = X_scaled_values[:,dim]
    y1 = Y_scaled_values
    pylab.scatter(x1, y1, alpha=0.5)
    pylab.show() # show the plot

# use dim=0 for Avg. Area House Age and dim=1 for Avg. Area Number of Rooms
interact(f, theta1=(0,1,0.1), theta0=(0,1,0.1), dim=(0,1,1))

def gd2(obsX, obsY, alpha, threshold):
    """
    Input: observed vectors X, Y, alpha and threshold
    Returns: theta0, theta1 from Gradient Descent L2 loss algorithm
             Iterations and L2 Error
    """
    theta0 = theta1 = 0
    oldError, newError = 0, 1
    iterations = 0
    while abs(newError - oldError) >= threshold:
        oldError = newError
        theta0 = theta0 - alpha * np.mean(h(theta0, theta1, obsX) - obsY)
        theta1 = theta1 - alpha * np.mean((h(theta0, theta1, obsX) - obsY) * obsX)
        newError = sqerror(obsX, obsY, theta0, theta1)
        iterations += 1
        print(f'Iteration {iterations}')
        print(f'theta0: {theta0}, theta1: {theta1}')
        print(f'error: {newError}')
    return [theta0, theta1, newError, iterations]

# Test the gd2 function
[theta0, theta1, newError, iterations] = gd2(x[:,[0]], y, 0.01, 0.0001)
print(iterations, newError, theta0, theta1)

def gdh(obsX, obsY, alpha, threshold, delta):
    """
    Input: observed vectors X, Y, alpha and threshold
    Returns: theta0, theta1 from Gradient Descent huber loss algorithm
             Iterations and huber Error
    """
    theta0 = theta1 = 0
    oldError, newError = 0, 1
    iterations = 0
    while abs(newError - oldError) >= threshold:
        oldError = newError
        theta0 = theta0 - alpha * np.mean(h(theta0, theta1, obsX) - obsY)
        theta1 = theta1 - alpha * np.mean((h(theta0, theta1, obsX) - obsY) * obsX)
        newError = huberror(obsX, obsY, theta0, theta1, delta)
        iterations += 1
        # print(f'Iteration {iterations}')
        # print(f'theta0: {theta0}, theta1: {theta1}')
        # print(f'error: {newError}')
    return [theta0, theta1, newError, iterations]

# Test the gdh function
[theta0, theta1, newError, iterations] = gdh(x[:,[0]], y, 0.01, 0.0001, 0.1)
print(iterations, newError, theta0, theta1)

lm = LinearRegression()
result = lm.fit(x[:,[0]].reshape(-1, 1),y)
theta0 = result.intercept_
theta1 = result.coef_
print(sqerror(x[:,0], y, theta0, theta1[0]))

def sqerror_multivariate(obsX, obsY, theta0, theta1, theta2):
    """
    Input: parameters theta0, theta1 and theta2 of the model
           x, y vectors
    Returns: L2 square error
    Assumptions: none
    """
    y_pred = theta0 + theta1 * obsX[:,0] + theta2 * obsX[:,1]
    squared_error = (y_pred - obsY)**2
    return squared_error.mean()

def gd22(obsX, obsY, alpha, threshold):
    """
    Input: observed vectors X, Y, alpha and threshold
    Returns: theta0, theta1, theta2 from Gradient Descent L2 loss algorithm
             Iterations and L2 Error
    """
    theta0 = theta1 = theta2 = 0
    oldError, newError = 0, 1
    iterations = 0
    while abs(newError - oldError) >= threshold:
        oldError = newError
        y_pred = theta0 + theta1 * obsX[:,0] + theta2 * obsX[:,1]
        theta0 = theta0 - alpha * np.mean(y_pred - obsY)
        theta1 = theta1 - alpha * np.mean((y_pred - obsY) * obsX[:,0])
        theta2 = theta2 - alpha * np.mean((y_pred - obsY) * obsX[:,1])
        newError = sqerror_multivariate(obsX, obsY, theta0, theta1, theta2)
        iterations += 1
        print(f'Iteration {iterations}')
        print(f'theta0: {theta0}, theta1: {theta1}, theta2: {theta2}')
        print(f'error: {newError}')
    return [theta0, theta1, theta2, newError, iterations]

# Test the gd22 function
[theta0, theta1, theta2, newError, iterations] = gd22(x, y, 0.01, 0.0001)
print(iterations, newError, theta0, theta1, theta2)

y = df_training.Price
X = df_training[["Avg. Area House Age", "Avg. Area Number of Rooms"]].values

model = LinearRegression().fit(X, y)
model
print('sklearn model:')
print(model.coef_)
print(model.intercept_)
print(metrics.mean_squared_error(y, model.predict(X)))

print('\ngradient descent model:')
with contextlib.redirect_stdout(StringIO()):
    [theta0, theta1, theta2, newError, iterations] = gd22(X, y, 0.01, 0.0001)
print([theta1, theta2], theta0, newError, sep='\n')

y_pred_gd = theta0 + theta1 * df_testing['Avg. Area House Age'] + theta2 * df_testing['Avg. Area Number of Rooms']
error_gd = np.mean(abs(y_pred_gd - df_testing['Price']))
print(error_gd)

y_pred_sklearn = model.predict(df_testing[["Avg. Area House Age", "Avg. Area Number of Rooms"]])
error_lib = np.mean(abs(y_pred_sklearn - df_testing['Price']))
print(error_lib)