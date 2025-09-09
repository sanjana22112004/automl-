import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def plot_corr():
    fig, ax = plt.subplots()
    corr = np.corrcoef(np.random.randn(10, 10))
    cax = ax.matshow(corr)
    plt.colorbar(cax)
    st.pyplot(fig)

def plot_roc_auc():
    from sklearn.metrics import roc_curve, auc
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:0.2f})")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    st.pyplot(fig)
