import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = []
    theta_history = [theta.copy()]  # Store initial parameters
    filenames = []  # To store filenames of the plots

    for i in range(num_iters):
        predictions = np.dot(X, theta)  # Calculate predictions
        errors = predictions - y
        gradient = (1/m) * np.dot(X.T, errors)  # Calculate gradient descent
        theta -= alpha * gradient  # Update parameters
        
        cost = mean_squared_error(y, predictions)  # Calculate MSE
        cost_history.append(cost)
        theta_history.append(theta.copy())  

        # Plotting step for each iteration
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 1], y, color='blue')  
        plt.plot(X[:, 1], predictions, color='red')  
        plt.title(f'Iteration {i+1}: Cost = {cost:.4f}')
        plt.xlabel('X')
        plt.ylabel('Y')
        filename = f'frame_{i}.png'
        plt.savefig(filename)  
        filenames.append(filename)  
        plt.close()  

    return theta, cost_history, theta_history, filenames

def create_gif(filenames, output_path='linear_regression.gif', duration=0.5):
    with imageio.get_writer(output_path, mode='I', duration=duration) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        for filename in filenames:  # Remove files after creating the GIF
            os.remove(filename)

# Setup the dataset
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]

# Initialize parameters
theta = np.random.randn(2, 1)

# Set hyperparameters
alpha = 0.1
num_iters = 25  

# Apply gradient descent
theta_optimized, cost_history, theta_history, filenames = gradient_descent(X_b, y, theta, alpha, num_iters)

# Create and save the GIF
create_gif(filenames)

# Optionally, plot the cost history
plt.figure()
plt.plot(range(num_iters), cost_history, '-o')
plt.title('Cost Function History')
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.savefig('cost-function-history')
plt.show()