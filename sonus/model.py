"""
Contains code to generate, train, test, and validate scikit-learn models.
Also contains code to load trained model from storage.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

def generate_pipeline(model):
    return Pipeline([("Scaler", StandardScaler()), ("Zero Var Remover", VarianceThreshold()), ("Model", model)])
