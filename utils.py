import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

def plot_corr_matrix(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        fig = plt.figure(figsize=(4,2))
        plt.text(0.5,0.5,"Not enough numeric cols", ha="center")
        return fig
    corr = num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6,5))
    cax = ax.imshow(corr, interpolation="nearest")
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns)
    fig.colorbar(cax)
    fig.tight_layout()
    return fig

def plot_conf_mat(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,4))
    disp = ConfusionMatrixDisplay(cm, display_labels=None)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig

def plot_roc_binary(y_true, y_proba):
    if y_proba is None:
        fig = plt.figure(); plt.text(0.5,0.5,"No predict_proba available", ha="center"); return fig
    if y_proba.ndim==2 and y_proba.shape[1]>1:
        y_scores = y_proba[:,1]
    else:
        y_scores = y_proba
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0,1],[0,1], linestyle="--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig

def plot_feature_importance(model, feature_names):
    if model is None or not hasattr(model, "feature_importances_"):
        fig = plt.figure(); plt.text(0.5,0.5,"No feature importances available", ha="center"); return fig
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    names = [feature_names[i] if i < len(feature_names) else str(i) for i in idx]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.barh(range(min(20,len(idx))), imp[idx][:20][::-1])
    ax.set_yticks(range(min(20,len(idx)))); ax.set_yticklabels(names[:20][::-1])
    ax.set_title("Feature Importances (top)")
    fig.tight_layout()
    return fig
