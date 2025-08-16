# Forest Cover Type Prediction with XGBoost
Predicts forest cover types using terrain features - **automatically downloads the dataset from sklearn on first run!**

This end-to-end ML pipeline demonstrates:
- Automatic dataset downloading
- Robust preprocessing with feature selection
- Hyperparameter tuning via RandomizedSearchCV
- Model evaluation metrics
- Feature importance visualization

## How It Works
The project uses sklearn's built-in `fetch_covtype` dataset that:
- Automatically downloads (~75MB) on first run
- Caches locally for future runs
- Requires no manual data uploads

## What's Inside
forest-cover-prediction/
├── src/                    # All source code
│   ├── data_loader.py      # Auto-downloads dataset
│   ├── preprocessor.py     # Feature engineering
│   ├── trainer.py          # Model training & tuning
│   └── evaluator.py        # Performance evaluation
├── config/                 # Centralized settings
│   └── params.yaml         # Tuning parameters
├── results/                # Output visualizations
├── tests/                  # Quality assurance
├── main.py                 # Run entire workflow
├── requirements.txt        # Python dependencies
└── README.md               # You're here!
