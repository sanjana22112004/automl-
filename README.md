
AutoML Phase 1 Demo (Tabular + Image)
====================================

This demo app lets users either:
- Choose from preloaded sample datasets (no API needed)
- Upload a CSV file
- Search & download from Kaggle (requires uploading your kaggle.json API token)

Included sample datasets: Titanic, Walmart sample, House Prices sample, IMDB sample, MNIST (image demo via HuggingFace).

How to run locally:
1. Create a virtual environment with Python 3.11
2. pip install -r requirements.txt
3. streamlit run app.py

Notes:
- Kaggle functionality requires a kaggle.json from your Kaggle account.
- Image pipeline is lightweight demo mode for Phase 1; full training can be added later.
