import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from src.config import load_config, get_path


def load_and_split_data():
    config = load_config()
    data_config = config['data']

    # Load dataset
    data = fetch_covtype()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target) - 1  # Adjust labels to 0-6

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_config['test_size'],
        stratify=y if data_config['stratify'] else None,
        random_state=data_config['random_state']
    )

    # Save data for reproducibility
    data_path = get_path('data')
    pd.concat([X, y], axis=1).to_csv(data_path, index=False)

    return X_train, X_test, y_train, y_test, data.feature_names
