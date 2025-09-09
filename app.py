import streamlit as st
import pandas as pd
import numpy as np
from datasets_search import search_datasets, get_sklearn_datasets, search_openml_datasets, load_dataset
from utils import plot_corr, plot_roc_auc, display_data_info, plot_data_distribution
from models import train_model
from preprocessing import preprocess

st.title("🤖 ML Streamlit Demo")
st.markdown("---")

# Sidebar for dataset selection
st.sidebar.title("📊 Dataset Selection")

# Dataset source selection
data_source = st.sidebar.selectbox(
    "Choose your data source:",
    ["Upload File", "Sklearn Datasets", "OpenML Datasets", "Kaggle Search"]
)

dataset = None
data_info = None

if data_source == "Upload File":
    st.sidebar.subheader("📁 Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with your data"
    )
    
    if uploaded_file is not None:
        try:
            dataset = pd.read_csv(uploaded_file)
            data_info = {
                'name': uploaded_file.name,
                'source': 'uploaded',
                'shape': dataset.shape
            }
            st.sidebar.success(f"✅ File uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {str(e)}")

elif data_source == "Sklearn Datasets":
    st.sidebar.subheader("🔬 Sklearn Datasets")
    sklearn_datasets = get_sklearn_datasets()
    
    selected_dataset = st.sidebar.selectbox(
        "Select a dataset:",
        list(sklearn_datasets.keys())
    )
    
    if st.sidebar.button("Load Dataset"):
        try:
            dataset, data_info = load_dataset('sklearn', selected_dataset)
            st.sidebar.success(f"✅ {selected_dataset} loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading dataset: {str(e)}")

elif data_source == "OpenML Datasets":
    st.sidebar.subheader("🌐 OpenML Datasets")
    search_query = st.sidebar.text_input("Search OpenML datasets:", placeholder="e.g. iris, wine, diabetes")
    
    if search_query:
        try:
            openml_results = search_openml_datasets(search_query)
            if openml_results:
                selected_openml = st.sidebar.selectbox(
                    "Select a dataset:",
                    [f"{result['name']} (ID: {result['id']})" for result in openml_results[:10]]
                )
                
                if st.sidebar.button("Load OpenML Dataset"):
                    dataset_id = int(selected_openml.split("ID: ")[1].split(")")[0])
                    dataset, data_info = load_dataset('openml', dataset_id)
                    st.sidebar.success(f"✅ OpenML dataset loaded successfully!")
            else:
                st.sidebar.warning("No datasets found for your search.")
        except Exception as e:
            st.sidebar.error(f"❌ Error searching OpenML: {str(e)}")

elif data_source == "Kaggle Search":
    st.sidebar.subheader("🏆 Kaggle Datasets")
    query = st.sidebar.text_input("Search Kaggle datasets", placeholder="e.g. Titanic, Iris, MNIST")
    if query:
        results = search_datasets(query)
        st.sidebar.write("Top Results:", results)

# Main content area
if dataset is not None:
    st.header("📈 Dataset Overview")
    
    # Display dataset information
    display_data_info(dataset, data_info)
    
    # Data preview
    st.subheader("🔍 Data Preview")
    st.dataframe(dataset.head(10))
    
    # Basic statistics
    st.subheader("📊 Basic Statistics")
    st.dataframe(dataset.describe())
    
    # Data visualization options
    st.subheader("📊 Visualizations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Show Correlation Matrix"):
            plot_corr(dataset)
    
    with col2:
        if st.button("Show ROC Curve"):
            plot_roc_auc(dataset)
    
    with col3:
        if st.button("Show Data Distribution"):
            plot_data_distribution(dataset)
    
    # Machine Learning section
    st.header("🤖 Machine Learning")
    
    if st.button("Train Model"):
        try:
            # Basic preprocessing
            processed_data = preprocess(dataset)
            
            # Train model (placeholder)
            model_result = train_model(processed_data, None)
            st.success(f"✅ {model_result}")
        except Exception as e:
            st.error(f"❌ Error training model: {str(e)}")

else:
    st.info("👈 Please select a dataset from the sidebar to get started!")
    
    # Show demo visualizations when no dataset is selected
    st.header("🎨 Demo Visualizations")
    st.write("Here are some sample visualizations:")
    plot_corr()
    plot_roc_auc()
