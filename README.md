# EnergyLens Streamlit App — Redesigned Version

This redesigned Streamlit app contains four required interfaces:

1. Global Dashboard
2. Country Explorer
3. Forecasting (LSTM + Prophet)
4. Cluster Analysis

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud deployment

Upload these files to a GitHub repository:

```text
app.py
requirements.txt
README.md
.streamlit/config.toml
```

Then deploy from Streamlit Community Cloud and set the main file path to:

```text
app.py
```

For best compatibility with Prophet and TensorFlow, select Python 3.11 if Streamlit Cloud asks for a Python version.
