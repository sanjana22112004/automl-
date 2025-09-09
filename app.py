import streamlit as st
import pandas as pd
import os
from datasets_search import SAMPLE_REGISTRY, load_sample, search_kaggle_and_download, requires_kaggle
from preprocessing import detect_task_and_preprocess
from models import train_and_select_model
from image_models import run_image_pipeline, is_image_hf_image_dataset
from utils import plot_corr_matrix, plot_conf_mat, plot_roc_binary, plot_feature_importance

st.set_page_config(page_title="AutoML — Demo (Tabular + Image)", layout="wide")
st.title("AutoML — Demo (Tabular + Image)")

st.sidebar.header("Dataset")
choice = st.sidebar.radio("Choose how to provide dataset", ["Pick a sample dataset", "Upload CSV", "Search Kaggle"])

df = None
selected_sample = None

if choice == "Pick a sample dataset":
    # show suggestions (autocomplete-like)
    sample_list = [f'{s["id"]}: {s["title"]} — {s["short"]}' for s in SAMPLE_REGISTRY]
    selected = st.sidebar.selectbox("Sample datasets (click to choose)", sample_list)
    idx = sample_list.index(selected)
    selected_sample = SAMPLE_REGISTRY[idx]
    st.sidebar.write("Description:", selected_sample["short"])
    if st.sidebar.button("Load sample dataset"):
        df = load_sample(selected_sample["id"])
        st.success(f"Loaded sample: {selected_sample['title']}")

elif choice == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("Uploaded CSV loaded")

else:  # Kaggle search
    st.sidebar.markdown("To search Kaggle, upload your `kaggle.json` (API token) first")
    kaggle_file = st.sidebar.file_uploader("Upload kaggle.json (optional)", type=["json"])
    query = st.sidebar.text_input("Search Kaggle (type keywords)")
    if st.sidebar.button("Search Kaggle"):
        if not kaggle_file:
            st.sidebar.warning("Please upload kaggle.json to use Kaggle search/download")
        else:
            try:
                credentials = json.load(kaggle_file)
                st.sidebar.success("kaggle.json uploaded (will be used for this session)")
                results = search_kaggle_and_download(query, credentials)
                if results and isinstance(results, dict) and results.get("local_csv"):
                    df = pd.read_csv(results["local_csv"])
                    st.success(f"Downloaded Kaggle dataset: {results.get('title','unknown')}")
                else:
                    st.sidebar.info("No CSV found in the downloaded Kaggle dataset or download failed.")
            except Exception as e:
                st.sidebar.error(f"Kaggle error: {e}")

# If dataset loaded and it's tabular:
if df is not None and not is_image_hf_image_dataset(df if isinstance(df, str) else None):
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
