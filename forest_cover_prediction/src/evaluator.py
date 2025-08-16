import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix, classification_report)
from src.config import get_path


def evaluate_model(model, X_test, y_test, feature_names):
    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro')
    }

    # Generate report
    report = classification_report(y_test, y_pred)

    # Generate visualizations
    results_path = get_path('results')

    # Confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(f"{results_path}/confusion_matrix.png")
    plt.close()

    # Feature importance (if available)
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        plt.figure(figsize=(12, 8))
        importances = model.named_steps['model'].feature_importances_

        # Use actual feature names
        features = feature_names

        sorted_idx = importances.argsort()
        plt.barh(range(len(sorted_idx[-20:])), importances[sorted_idx][-20:])
        plt.yticks(range(len(sorted_idx[-20:])), [features[i] for i in sorted_idx[-20:]])
        plt.title('Top 20 Feature Importances')
        plt.savefig(f"{results_path}/feature_importance.png")
        plt.close()

    return metrics, report