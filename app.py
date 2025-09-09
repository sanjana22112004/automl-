import streamlit as st
import pandas as pd
import os
from preprocessing import detect_task_and_preprocess
from models import train_and_select_model
from utils import plot_corr_matrix, plot_conf_mat, plot_roc_binary, plot_feature_importance

st.set_page_config(page_title="AutoML — Demo (Tabular + Image)", layout="wide")
st.title("AutoML — Demo (Tabular + Image)")

st.sidebar.header("Dataset")
st.sidebar.write("Upload a CSV file to begin.")

df = None
uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.success("Uploaded CSV loaded")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

# If dataset loaded:
if df is not None:
    st.subheader("Dataset preview")
    st.dataframe(df.head(50))
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")

    # Correlation (raw numeric)
    st.caption("Correlation (numeric features)")
    try:
        fig_corr = plot_corr_matrix(df)
        st.pyplot(fig_corr)
    except Exception as e:
        st.write("Correlation plot failed:", e)

    # Detect task and preprocess, suggest target
    task_type, X_train, X_test, y_train, y_test, suggested_target, feature_names, cleaned_df = detect_task_and_preprocess(df)
    st.info(f"Suggested target column: `{suggested_target}` (please confirm below)")
    target_confirm = st.text_input("Confirm target column (type exact column name):", value=suggested_target)

    if st.button("Run tabular AutoML"):
        try:
            task_type, X_train, X_test, y_train, y_test, _, feature_names, cleaned_df = detect_task_and_preprocess(df, target_confirm)
            result = train_and_select_model(task_type, X_train, X_test, y_train, y_test)
            st.success(f"Best model: {result['best_model_name']} — {result['metric_name']}: {result['score']:.4f}")

            if task_type == "classification":
                st.caption("Confusion matrix")
                st.pyplot(plot_conf_mat(y_test, result["y_pred"]))
                # ROC if binary
                if len(pd.Series(y_test).unique()) == 2:
                    st.caption("ROC curve")
                    st.pyplot(plot_roc_binary(y_test, result.get("y_proba")))
            else:
                st.caption("Feature importances (if model supports it)")
                st.pyplot(plot_feature_importance(result.get("model"), feature_names))
        except Exception as e:
            st.error(f"Training failed: {e}")

# If an HF image dataset name string was returned (special flow)
# Note: in this simplified demo, image pipeline is triggered directly via sample selection for MNIST
