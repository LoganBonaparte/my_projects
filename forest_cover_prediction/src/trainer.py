import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from src.config import load_config, get_path
from src.preprocessor import build_preprocessing_pipeline


def train_model(X_train, y_train):
    config = load_config()
    model_config = config['model']
    training_config = config['training']

    # Build pipeline
    preprocessing = build_preprocessing_pipeline()

    model = XGBClassifier(
        objective=model_config['objective'],
        eval_metric=model_config['eval_metric'],
        num_class=model_config['num_class'],
        random_state=model_config['random_state'],
        n_jobs=training_config['n_jobs']
    )

    pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('model', model)
    ])

    # Parameter grid
    param_grid = {
        'preprocessing__selector__k': [20, 30, 40],
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [4, 6, 8],
        'model__learning_rate': [0.05, 0.1, 0.2],
        'model__subsample': [0.7, 0.8, 1.0],
        'model__colsample_bytree': [0.7, 0.8, 1.0],
        'model__reg_alpha': [0, 0.1, 0.5],
        'model__reg_lambda': [1, 1.5, 2]
    }

    # Hyperparameter tuning
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=training_config['n_iter'],
        cv=training_config['cv'],
        scoring=training_config['scoring'],
        n_jobs=training_config['n_jobs'],
        random_state=model_config['random_state']
    )

    search.fit(X_train, y_train)

    # Save best model
    model_path = get_path('model')
    joblib.dump(search.best_estimator_, model_path)

    return search.best_estimator_