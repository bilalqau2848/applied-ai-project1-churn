import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("Telco Customer Churn Prediction")
st.markdown("Lab 4 - From Notebook to Product")

@st.cache_resource
def load_artifacts():
    model = joblib.load('churn_model.joblib')
    with open('model_meta.json', 'r') as f:
        meta = json.load(f)
    sample = pd.read_csv('sample_customers.csv')
    return model, meta, sample

model, meta, sample_df = load_artifacts()

col1, col2, col3 = st.columns(3)
col1.metric("Model", meta['model_name'])
col2.metric("AUC", meta['metrics']['auc'])
col3.metric("Threshold", meta['threshold'])

st.divider()
st.subheader("Sample Customers Data")
st.dataframe(sample_df.head())

if st.button("Predict Churn for Sample Customers"):
    preds = model.predict(sample_df)
    probs = model.predict_proba(sample_df)[:, 1]
    
    result_df = sample_df.copy()
    result_df['Churn_Probability'] = probs
    result_df['Prediction'] = ['Churn' if p > meta['threshold'] else 'No Churn' for p in probs]
    
    st.subheader("Results")
    st.dataframe(result_df)
    
    churn_count = sum(preds)
    st.warning(f"Total Churn Predicted: {churn_count} out of {len(preds)} customers")

st.sidebar.info("Model files:\n- churn_model.joblib\n- model_meta.json\n- sample_customers.csv")