import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Generate synthetic dataset of systems of linear equations Ax = b
def generate_data(num_samples=10000, matrix_size=3):
    A = np.random.randn(num_samples, matrix_size, matrix_size)  # Random matrix A
    b = np.random.randn(num_samples, matrix_size)  # Random vector b
    x = np.linalg.solve(A, b.T).T  # Solve for x using numpy's solver
    return A, b, x

# Generate training data
num_samples = 10000
matrix_size = 3
A_train, b_train, x_train = generate_data(num_samples, matrix_size)

# Build the deep learning model
model = keras.Sequential([
    layers.InputLayer(input_shape=(matrix_size, matrix_size)),  # Input is the matrix A
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(matrix_size)  # Output is the solution vector x
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(A_train, x_train, epochs=10, batch_size=32)

# Evaluate the model
A_test, b_test, x_test = generate_data(1000, matrix_size)
loss = model.evaluate(A_test, x_test)
print(f"Test Loss: {loss}")

# Predict using the trained model
predictions = model.predict(A_test)

# Compare predicted vs actual solutions
for i in range(5):
    print(f"Predicted: {predictions[i]}, Actual: {x_test[i]}")

# Plot some predictions
plt.figure(figsize=(10, 6))
plt.plot(predictions[:5], label='Predicted')
plt.plot(x_test[:5], label='Actual', linestyle='dashed')
plt.legend()
plt.title("Predictions vs Actual Solutions")
plt.show()
