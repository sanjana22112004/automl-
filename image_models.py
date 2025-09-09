import streamlit as st
from datasets import load_dataset
import pandas as pd

def is_image_huggingface_dataset(obj):
    # If obj is a string with dataset name, treat as HF name; or if df-like, return False
    return isinstance(obj, str) and obj.lower() in ["mnist","cifar10","fashion_mnist","kmnist"]

def run_image_pipeline(name):
    st.info(f"Image demo: loading a small subset of {name}")
    try:
        ds = load_dataset(name, split="train[:200]")
        st.write("First 3 items:")
        try:
            st.write(ds.select(range(3)))
        except Exception:
            st.write(ds[:3])
        st.success("Image demo loaded (full training is for Phase 2).")
    except Exception as e:
        st.error(f"Failed to load HF image dataset: {e}")
