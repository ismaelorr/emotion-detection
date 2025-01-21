from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

class ModelTrainer:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.model = RandomForestClassifier()
    def train(self):
        self.model.fit(self.features, self.labels)
        print("Model training correctly.")
    def save_model(self, path):
        joblib.dump(self.model, path)
        print(f"Model save in {path}.")
    def evaluate(self):
        predictions = self.model.predict(self.features)
        print("Model evaluation:")
        print(classification_report(self.labels, predictions))
