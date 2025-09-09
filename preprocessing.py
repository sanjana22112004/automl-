import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def _is_classification_target(y: pd.Series) -> bool:
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        return True
    unique_vals = pd.unique(y.dropna())
    return len(unique_vals) <= 10

def detect_task_and_preprocess(df: pd.DataFrame, target=None, test_size=0.2, random_state=42):
    if target is None or target not in df.columns:
        target = df.columns[-1]
    df = df.copy()
    # drop columns with all missing
    df = df.dropna(axis=1, how='all')
    y = df[target]
    X = df.drop(columns=[target])

    task_type = "classification" if _is_classification_target(y) else "regression"

    cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    # OneHotEncoder kwargs compatibility
    import sklearn
    ohe_kwargs = {"handle_unknown":"ignore"}
    if sklearn.__version__ >= "1.2":
        ohe_kwargs["sparse_output"] = False
    else:
        ohe_kwargs["sparse"] = False

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_imputer, num_cols),
        ("cat", OneHotEncoder(**ohe_kwargs), cat_cols)
    ], remainder="drop")

    # Prefer stratified split for classification, but fall back if it's not feasible
    stratify_arg = y if task_type == "classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arg
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=None
        )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    feature_names = list(num_cols)
    if cat_cols:
        try:
            ohe = preprocessor.named_transformers_["cat"]
            if hasattr(ohe, "get_feature_names_out"):
                feature_names.extend(ohe.get_feature_names_out(cat_cols).tolist())
        except Exception:
            pass

    return task_type, X_train_t, X_test_t, y_train.values if hasattr(y_train,"values") else y_train, y_test.values if hasattr(y_test,"values") else y_test, target, feature_names, df, preprocessor, num_cols, cat_cols
