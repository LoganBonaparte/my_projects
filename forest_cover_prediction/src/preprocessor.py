from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

def build_preprocessing_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('variance_threshold', VarianceThreshold()),  # Remove constant features
        ('scaler', MinMaxScaler()),
        ('selector', SelectKBest(score_func=f_classif))
    ])