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
    # High-level dataset KPIs
    c1, c2 = st.columns(2)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")

    # Tabs for a cleaner layout
    tab_preview, tab_corr = st.tabs(["Preview", "Correlation"])
    with tab_preview:
        st.subheader("Dataset preview")
        st.dataframe(df.head(50))
        st.caption("Showing first 50 rows")
    with tab_corr:
        st.caption("Correlation (numeric features)")
        try:
            fig_corr = plot_corr_matrix(df)
            st.pyplot(fig_corr)
        except Exception as e:
            st.write("Correlation plot failed:", e)

    

    # Detect task and preprocess, suggest target
    task_type, X_train, X_test, y_train, y_test, suggested_target, feature_names, cleaned_df = detect_task_and_preprocess(df)
    st.info(f"Suggested target column: `{suggested_target}` (please confirm below)")
    # Show detected problem type and candidate algorithms clearly on screen
    if task_type == "classification":
        st.success("Detected task: Classification — candidate algorithms: LogisticRegression, RandomForestClassifier")
    else:
        st.success("Detected task: Regression — candidate algorithms: LinearRegression, RandomForestRegressor")
    target_confirm = st.text_input("Confirm target column (type exact column name):", value=suggested_target)

    if st.button("Run tabular AutoML"):
        try:
            task_type, X_train, X_test, y_train, y_test, _, feature_names, cleaned_df = detect_task_and_preprocess(df, target_confirm)
            result = train_and_select_model(task_type, X_train, X_test, y_train, y_test)
            st.subheader(f"Selected algorithm: {result['best_model_name']}")
            st.success(f"{result['metric_name']}: {result['score']:.4f}")
            st.caption("Chosen because it achieved the best validation score among candidates.")
            st.write(f"Selected model class: `{type(result['model']).__name__}`")

            # Show all model scores inline for quick comparison
            try:
                scores_df = pd.DataFrame(result["results"]).T
                st.caption("Model comparison")
                st.dataframe(scores_df)
            except Exception:
                st.write(result["results"])

            # Keep hyperparameters in an expander
            with st.expander("Model hyperparameters"):
                try:
                    st.json(result["model"].get_params())
                except Exception:
                    st.write("Parameters unavailable for this model.")

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
