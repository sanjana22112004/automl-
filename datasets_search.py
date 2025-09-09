import pandas as pd
import numpy as np
from sklearn import datasets
import openml
from typing import Dict, List, Tuple, Any

def search_datasets(query):
    """Search Kaggle datasets (placeholder implementation)"""
    # Dummy Kaggle dataset search
    return [f"{query} Dataset 1", f"{query} Dataset 2"]

def get_sklearn_datasets() -> Dict[str, Any]:
    """Get available sklearn datasets"""
    sklearn_datasets = {
        'Iris': datasets.load_iris,
        'Wine': datasets.load_wine,
        'Breast Cancer': datasets.load_breast_cancer,
        'Diabetes': datasets.load_diabetes,
        'Boston Housing': datasets.load_boston,
        'Digits': datasets.load_digits,
        'Linnerud': datasets.load_linnerud,
        'California Housing': datasets.fetch_california_housing,
        'Covtype': datasets.fetch_covtype,
        'Kddcup 99': datasets.fetch_kddcup99,
        'RCV1': datasets.fetch_rcv1,
        '20newsgroups': datasets.fetch_20newsgroups,
        'Olivetti Faces': datasets.fetch_olivetti_faces,
        'LFW People': datasets.fetch_lfw_people,
        'LFW Pairs': datasets.fetch_lfw_pairs,
    }
    return sklearn_datasets

def search_openml_datasets(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search OpenML datasets by query"""
    try:
        # Search for datasets
        datasets_list = openml.datasets.list_datasets(output_format='dataframe')
        
        # Filter by query (case insensitive)
        filtered_datasets = datasets_list[
            datasets_list['name'].str.contains(query, case=False, na=False)
        ]
        
        # Sort by number of instances (descending) and take top results
        filtered_datasets = filtered_datasets.sort_values('NumberOfInstances', ascending=False)
        
        # Convert to list of dictionaries
        results = []
        for idx, row in filtered_datasets.head(limit).iterrows():
            results.append({
                'id': int(row['did']),
                'name': row['name'],
                'instances': int(row['NumberOfInstances']),
                'features': int(row['NumberOfFeatures']),
                'classes': int(row['NumberOfClasses']) if pd.notna(row['NumberOfClasses']) else None
            })
        
        return results
    except Exception as e:
        print(f"Error searching OpenML datasets: {e}")
        return []

def load_dataset(source: str, dataset_identifier: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load dataset from specified source"""
    try:
        if source == 'sklearn':
            return load_sklearn_dataset(dataset_identifier)
        elif source == 'openml':
            return load_openml_dataset(dataset_identifier)
        else:
            raise ValueError(f"Unknown source: {source}")
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")

def load_sklearn_dataset(dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load sklearn dataset"""
    sklearn_datasets = get_sklearn_datasets()
    
    if dataset_name not in sklearn_datasets:
        raise ValueError(f"Dataset {dataset_name} not found")
    
    # Load the dataset
    dataset_func = sklearn_datasets[dataset_name]
    data = dataset_func()
    
    # Convert to DataFrame
    if hasattr(data, 'data') and hasattr(data, 'target'):
        # For datasets with features and targets
        df = pd.DataFrame(data.data, columns=data.feature_names if hasattr(data, 'feature_names') else [f'feature_{i}' for i in range(data.data.shape[1])])
        if hasattr(data, 'target'):
            df['target'] = data.target
    else:
        # For other types of datasets
        df = pd.DataFrame(data)
    
    # Create data info
    data_info = {
        'name': dataset_name,
        'source': 'sklearn',
        'shape': df.shape,
        'description': data.DESCR if hasattr(data, 'DESCR') else f"Sklearn {dataset_name} dataset"
    }
    
    return df, data_info

def load_openml_dataset(dataset_id: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load OpenML dataset by ID"""
    try:
        # Load dataset from OpenML
        dataset = openml.datasets.get_dataset(dataset_id)
        
        # Get the data
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute,
            return_categorical_indicator=True,
            return_attribute_names=True
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(X, columns=attribute_names)
        
        # Add target if available
        if y is not None:
            df['target'] = y
        
        # Create data info
        data_info = {
            'name': dataset.name,
            'source': 'openml',
            'shape': df.shape,
            'description': dataset.description,
            'dataset_id': dataset_id,
            'url': dataset.url
        }
        
        return df, data_info
        
    except Exception as e:
        raise Exception(f"Error loading OpenML dataset {dataset_id}: {str(e)}")
