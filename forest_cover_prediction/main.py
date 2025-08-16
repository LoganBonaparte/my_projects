from src.data_loader import load_and_split_data
from src.trainer import train_model
from src.evaluator import evaluate_model
import joblib
import os


def main():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Load and split data
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, feature_names = load_and_split_data()

    # Train model
    print("Training model...")
    model = train_model(X_train, y_train)

    # Evaluate model
    print("Evaluating model...")
    metrics, report = evaluate_model(model, X_test, y_test, feature_names)

    # Print results
    print("\n===== Evaluation Results =====")
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

    print("\nClassification Report:")
    print(report)

    print("\nResults saved in 'results/' directory")


if __name__ == "__main__":
    main()