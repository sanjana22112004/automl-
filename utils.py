import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

def display_data_info(dataset, data_info):
    """Display dataset information"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Dataset Name", data_info.get('name', 'Unknown'))
    
    with col2:
        st.metric("Shape", f"{data_info.get('shape', (0, 0))[0]} rows × {data_info.get('shape', (0, 0))[1]} columns")
    
    with col3:
        st.metric("Source", data_info.get('source', 'Unknown').title())
    
    # Display description if available
    if 'description' in data_info and data_info['description']:
        with st.expander("Dataset Description"):
            st.write(data_info['description'])
    
    # Display additional info for OpenML datasets
    if data_info.get('source') == 'openml' and 'url' in data_info:
        st.write(f"🔗 [View on OpenML]({data_info['url']})")

def plot_corr(dataset=None):
    """Plot correlation matrix"""
    if dataset is not None:
        # Select only numeric columns
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = dataset[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax, fmt='.2f')
            ax.set_title('Correlation Matrix')
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns for correlation matrix")
    else:
        # Demo correlation matrix
        fig, ax = plt.subplots()
        corr = np.corrcoef(np.random.randn(10, 10))
        cax = ax.matshow(corr)
        plt.colorbar(cax)
        ax.set_title('Demo Correlation Matrix')
        st.pyplot(fig)

def plot_roc_auc(dataset=None):
    """Plot ROC curve"""
    if dataset is not None:
        # Check if dataset has a target column
        if 'target' in dataset.columns:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            
            # Prepare data
            X = dataset.select_dtypes(include=[np.number]).drop('target', axis=1, errors='ignore')
            y = dataset['target']
            
            # Handle non-numeric targets
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            if len(X.columns) > 0 and len(np.unique(y)) == 2:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                # Train a simple classifier
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_train, y_train)
                
                # Get prediction probabilities
                y_scores = clf.predict_proba(X_test)[:, 1]
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:0.2f})')
                ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve')
                ax.legend(loc="lower right")
                st.pyplot(fig)
            else:
                st.warning("ROC curve requires binary classification with numeric features")
        else:
            st.warning("Dataset must have a 'target' column for ROC curve")
    else:
        # Demo ROC curve
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:0.2f})")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Demo ROC Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)

def plot_data_distribution(dataset):
    """Plot data distribution for numeric columns"""
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                axes[i].hist(dataset[col].dropna(), bins=30, alpha=0.7)
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No numeric columns found for distribution plot")
