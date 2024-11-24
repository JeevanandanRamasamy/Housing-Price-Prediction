# Housing Price Prediction

This script performs a multivariate linear regression analysis on a housing dataset, using both gradient descent and scikit-learn’s LinearRegression to predict housing prices. The script demonstrates the full process from data loading, preprocessing, and feature scaling, to building and evaluating regression models using different loss functions (L2, L1, and Huber). Interactive visualizations and error metric calculations are included to help understand model performance.

## Requirements

This project requires several Python libraries:
- pandas
- numpy
- seaborn
- matplotlib
- sklearn
- ipywidgets

To install the required dependencies, you can use the following command:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn ipywidgets
```

## Dataset

The dataset used is the `USA_Housing.csv`, which contains housing data for the USA. The key features include:
- `Avg. Area House Age`: The average age of houses in the area
- `Avg. Area Number of Rooms`: The average number of rooms in the houses
- `Price`: The price of the house

---

## Steps & Functions

### 1. Data Preprocessing
- Load data from a CSV file.
- Select relevant features for modeling (`Avg. Area House Age`, `Avg. Area Number of Rooms`, and `Price`).
- Split data into training and testing sets.
- Normalize features using **MinMaxScaler**.

### 2. Visualization
- Scatter plots of `Avg. Area House Age` vs. `Price` and `Avg. Area Number of Rooms` vs. `Price`.
- Interactive visualization using **ipywidgets** to adjust the regression line parameters (`theta0`, `theta1`) for each feature (`Avg. Area House Age`, `Avg. Area Number of Rooms`).

### 3. Error Calculation Functions
- **L2 (Squared Error)**: Measures the mean squared difference between predicted and actual values.
-	**L1 (Absolute Error)**: Measures the mean absolute difference between predicted and actual values.
-	**Huber Error**: A combination of L1 and L2 error to make the model more robust to outliers.

### 4. Gradient Descent
- **Gradient Descent for L2 Loss**: The gd2 function implements gradient descent for minimizing the L2 loss to find the optimal model parameters (`theta0`, `theta1`).
-	**Gradient Descent for Huber Loss**: The gdh function uses the Huber loss, which is more robust to outliers, to find the optimal model parameters.

### 5. Model Fitting and Evaluation
-	**Linear Regression Model (scikit-learn)**: Fits a linear regression model using `LinearRegression` from scikit-learn.
-	**Gradient Descent Model**: A custom implementation of linear regression using gradient descent for both L2 and Huber loss.
-	Comparison of the models using the **mean squared error** and the **mean absolute error**.

### 6. Comparison of Gradient Descent and Scikit-learn Models
- After training both models, the script evaluates their performance on the test set by calculating the mean absolute error for both models.

**Key Functions**
-	`sqerror(x, y, theta0, theta1)`: Computes the squared error (L2 loss) between predicted and actual values.
-	`abserror(x, y, theta0, theta1)`: Computes the absolute error (L1 loss).
-	`huberror(x, y, theta0, theta1, delta)`: Computes the Huber error.
-	`gd2(obsX, obsY, alpha, threshold)`: Performs gradient descent with L2 loss.
-	`gdh(obsX, obsY, alpha, threshold, delta)`: Performs gradient descent with Huber loss.
-	`gd22(obsX, obsY, alpha, threshold)`: Performs multivariate gradient descent for L2 loss with two input features.
