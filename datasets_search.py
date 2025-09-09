import os, tempfile, json, subprocess, zipfile, glob
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

# Kaggle search+download helper (requires kaggle.json content as dict)
def search_kaggle_and_download(query, kaggle_credentials):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        return None
    # write credentials temporarily
    td = tempfile.mkdtemp()
    kdir = os.path.join(td, ".kaggle")
    os.makedirs(kdir, exist_ok=True)
    with open(os.path.join(kdir, "kaggle.json"), "w") as f:
        json.dump(kaggle_credentials, f)
    os.environ["KAGGLE_CONFIG_DIR"] = kdir
    api = KaggleApi()
    api.authenticate()
    # search datasets
    res = api.datasets_list(search=query, page=1, max_results=5)
    if not res:
        return None
    # pick first result and download
    ds = res[0]
    ref = ds.ref  # owner/dataset-name
    td2 = tempfile.mkdtemp()
    api.dataset_download_files(ref, path=td2, unzip=True)
    # find first csv
    files = glob.glob(os.path.join(td2, "**", "*.csv"), recursive=True)
    if not files:
        return {"title": ds.title, "local_csv": None}
    return {"title": ds.title, "local_csv": files[0]}
