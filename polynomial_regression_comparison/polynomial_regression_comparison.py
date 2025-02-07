import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Generate synthetic noisy data
np.random.seed(0)
def generate_data(degree=3, num_points=100, noise_level=10):
    X = np.linspace(-5, 5, num_points).reshape(-1, 1)
    # Create a random polynomial and evaluate it at X
    coeff = np.random.rand(degree + 1)
    y = np.poly1d(coeff)(X.flatten())
    # Add Gaussian noise to make the problem realistic
    y += np.random.normal(scale=noise_level, size=y.shape)
    return X, y

# Neural network model for polynomial fitting
def build_nn_model(input_dim=1):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Traditional polynomial fitting using sklearn
def fit_polynomial(X, y, degree=3):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    return model, poly

# Function to evaluate and plot the model results
def plot_results(X, y, nn_pred, poly_pred, title="Polynomial Regression Comparison"):
    plt.scatter(X, y, color='black', label='Data', s=10)
    plt.plot(X, nn_pred, color='blue', label='AI Model', linewidth=2)
    plt.plot(X, poly_pred, color='red', label='Polynomial Fit', linewidth=2)
    plt.legend()
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

# Main function to generate data, train models and evaluate results
def main():
    # Generate synthetic noisy data
    X, y = generate_data(degree=3, num_points=100, noise_level=10)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # --- Train AI-based neural network model ---
    nn_model = build_nn_model(input_dim=X.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    nn_model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)
    
    # Get predictions from the neural network
    nn_pred = nn_model.predict(X_test)
    
    # --- Train traditional polynomial regression model ---
    poly_model, poly = fit_polynomial(X_train, y_train, degree=3)
    X_test_poly = poly.transform(X_test)
    poly_pred = poly_model.predict(X_test_poly)
    
    # --- Evaluate the models ---
    nn_mse = mean_squared_error(y_test, nn_pred)
    poly_mse = mean_squared_error(y_test, poly_pred)
    
    print(f"AI-based Model (NN) Mean Squared Error: {nn_mse:.4f}")
    print(f"Traditional Polynomial Regression Mean Squared Error: {poly_mse:.4f}")
    
    # --- Plot Results ---
    plot_results(X_test, y_test, nn_pred.flatten(), poly_pred.flatten())

if __name__ == "__main__":
    main()
