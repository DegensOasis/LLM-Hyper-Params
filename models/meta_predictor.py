import joblib
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class MetaPredictor:
    def __init__(self):
        self.model = RandomForestRegressor()
        self.feature_names = None

    def train(self, X, y):
        self.feature_names = X.columns if hasattr(X, 'columns') else None
        self.model.fit(X, y)

    def predict(self, prompt_features):
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(prompt_features)

    def save(self, filename):
        joblib.dump((self.model, self.feature_names), filename)

    @classmethod
    def load(cls, filename):
        predictor = cls()
        predictor.model, predictor.feature_names = joblib.load(filename)
        return predictor

def extract_prompt_features(prompt):
    return np.array([[
        len(prompt),
        len(prompt.split()),
        prompt.count('.') + prompt.count('!') + prompt.count('?'),
        len(set(prompt.split())),
        sum(1 for c in prompt if c.isupper()) / len(prompt),
        sum(1 for c in prompt if c.isdigit()) / len(prompt),
    ]])
