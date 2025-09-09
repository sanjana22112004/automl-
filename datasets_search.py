import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)

# Sample dataset registry (id, title, short description, local path)
SAMPLE_REGISTRY = [
    {"id":"titanic", "title":"Titanic - Passenger Survival", "short":"Predict survival of passengers (classification)", "local":"samples/titanic.csv"},
    {"id":"walmart", "title":"Walmart Sales (sample)", "short":"Weekly store sales data (regression demo)", "local":"samples/walmart_sample.csv"},
    {"id":"house_prices", "title":"House Prices - Ames (sample)", "short":"Housing features to predict sale price (regression)", "local":"samples/house_prices_sample.csv"},
    {"id":"imdb", "title":"IMDB Reviews (sample)", "short":"Movie reviews with sentiment labels (text classification)", "local":"samples/imdb_sample.csv"},
    {"id":"mnist", "title":"MNIST (image)", "short":"Handwritten digits (image classification)", "local":None}
]

def load_sample(sample_id):
    ent = next((s for s in SAMPLE_REGISTRY if s["id"]==sample_id), None)
    if ent is None:
        raise ValueError("Unknown sample id")
    if ent["local"] is None:
        # For MNIST, return dataset name so app can handle it as image dataset
        return ent["id"]
    path = os.path.join(os.path.dirname(__file__), ent["local"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sample not found: {path}")
    return pd.read_csv(path)

# Kaggle functionality removed.
