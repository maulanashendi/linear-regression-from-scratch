# Linear Regression From Scratch
## Overview
This repository contains Python code that demonstrates how to implement linear regression using gradient descent from scratch. Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. The goal is to find a linear relationship between the independent variable(s) and the dependent variable.


## Key Concepts
### Linear Regression
Linear regression is a statistical method used to understand the relationship between two variables: one independent (explanatory) variable and one dependent variable (the one you want to predict). The goal is to be able to predict the value of the dependent variable based on the value of the independent variable. the model looks like this:

```math
y = \beta_0 + \beta_1 x + \epsilon
```

### Cost Function
The cost function, also known as the loss function, measures the performance of the machine learning model. For linear regression, we often use the Mean Squared Error (MSE) which calculates the average squared difference between the predicted and actual values:

```math
MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
```

### Gradient Descent
Gradient descent is an optimization algorithm used to minimize the cost function. It iteratively adjusts the parameters beta_0, beta_1 to find the best line that fits the data. This is achieved by taking the derivative of the cost function to find the direction to move the parameters to reduce the error.

```math
\theta_{\text{new}} = \theta_{\text{old}} - \alpha \cdot \nabla J(\theta)
```
  
## Implementation
The Python script `linear_regression.py` includes the following components:
- A function `mean_squared_error` to calculate the MSE.
- A function `gradient_descent` to perform the gradient descent optimization.
- A function `create_gif` to generate gif every iteration(for fun).

## Output
The final output includes a scatter plot of the original data and lines representing the linear regression model at each iteration of gradient descent, showing how the model converges to the best fit.

![](https://github.com/maulanashendi/linear-regression-from-scratch/blob/main/linear_regression.gif) 
![](https://github.com/maulanashendi/linear-regression-from-scratch/blob/main/cost-function-history.png)
