import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

def _rmse(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

def train_and_select_model(task_type, X_train, X_test, y_train, y_test):
    results = {}
    trained = {}
    if task_type == "classification":
        candidates = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42)
        }
        best_name, best_score = None, -1
        for name, model in candidates.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")
            score = (acc + f1) / 2
            results[name] = {"accuracy": float(acc), "f1": float(f1)}
            trained[name] = model
            if score > best_score:
                best_score = score; best_name = name
        return {"best_model_name": best_name, "metric_name":"accuracy/f1", "score":best_score, "model":trained[best_name], "y_pred":trained[best_name].predict(X_test), "y_proba":trained[best_name].predict_proba(X_test) if hasattr(trained[best_name], "predict_proba") else None, "results":results}
    else:
        candidates = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42)
        }
        best_name, best_score = None, 1e18
        for name, model in candidates.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = _rmse(y_test, preds)
            results[name] = {"rmse": float(rmse)}
            trained[name] = model
            if rmse < best_score:
                best_score = rmse; best_name = name
        return {"best_model_name": best_name, "metric_name":"RMSE", "score":best_score, "model":trained[best_name], "y_pred":trained[best_name].predict(X_test), "y_proba":None, "results":results}
