import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Step 1: Create synthetic nonlinear optimization problems

# Define a complex objective function (nonlinear)
def objective_function(x):
    return np.sin(x[0])**2 + np.cos(x[1])**2 + 0.1 * (x[0] - 2)**2 + 0.2 * (x[1] + 3)**2

# Generate data (random initial points) for training
def generate_data(num_samples=1000, num_variables=2):
    X = np.random.uniform(-10, 10, (num_samples, num_variables))  # Random initial points
    y = np.array([objective_function(x) for x in X])  # Evaluate the objective function at those points
    return X, y

# Step 2: Build the neural network model

def build_model(input_dim):
    model = keras.Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Output layer for predicting the objective function value
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Step 3: Train the model
input_dim = 2  # Number of variables in the optimization problem
X_train, y_train = generate_data(num_samples=5000, num_variables=input_dim)

model = build_model(input_dim)
model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=0)

# Step 4: Evaluate the trained model on unseen data
X_test, y_test = generate_data(num_samples=1000, num_variables=input_dim)
predictions = model.predict(X_test)

# Visualize the predictions vs actual values
plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions.flatten(), cmap='viridis')
plt.colorbar(label='Predicted Objective Value')
plt.title('Predicted Objective Values from Neural Network')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# Step 5: Use Gradient Descent to find the true optimal solution
def gradient_descent_optimization():
    initial_guess = np.random.uniform(-10, 10, input_dim)
    result = minimize(objective_function, initial_guess, method='BFGS')
    return result.x, result.fun

# Step 6: Compare predictions with gradient descent solution
gradient_solution, gradient_value = gradient_descent_optimization()

print(f"Gradient Descent Optimal Solution: {gradient_solution}")
print(f"Objective Function Value (Gradient Descent): {gradient_value}")

# Find predicted optimal solution using the trained neural network
predicted_solution = model.predict(gradient_solution.reshape(1, -1))

print(f"Predicted Optimal Solution from Neural Network: {gradient_solution}")
print(f"Predicted Objective Value (Neural Network): {predicted_solution.flatten()[0]}")

# Step 7: Performance comparison
print("\nPerformance Comparison:")
print(f"Objective Function Value (True Optimal): {gradient_value}")
print(f"Objective Function Value (Predicted from NN): {predicted_solution.flatten()[0]}")
