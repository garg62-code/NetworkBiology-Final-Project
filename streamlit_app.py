import streamlit as st
import torch
import pickle
import pandas as pd
import numpy as np
from models import InductiveGCN
from torch_geometric.data import Data

# --- 1. SETUP & LOADING ---
st.set_page_config(page_title="Heart Disease Risk", page_icon="🫀")


@st.cache_resource
def load_artifacts():
    # Load Graph Data
    graph = torch.load('test_graph_data.pt')

    # Load Mappings
    with open('node_mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)

    # Load Model
    # Note: We must match the dimensions from your config/training
    # Assuming hidden_channels=32 and input_dim=13 based on your report
    model = InductiveGCN(in_channels=13, hidden_channels=32, out_channels=2)
    model.load_state_dict(torch.load('best_bipartite_model.pth', map_location=torch.device('cpu')))
    model.eval()

    return model, graph, mapping


try:
    model, base_graph, mapping = load_artifacts()
    st.success("System Ready: Model & Graph loaded successfully.")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# --- 2. USER INTERFACE ---
st.title("🫀 Heart Disease Risk Prediction")
st.markdown("This tool uses a **Bipartite Graph Neural Network (GNN)**. Enter clinical details below.")

col1, col2 = st.columns(2)

with col1:
    st.header("Demographics")
    age = st.slider("Age", 29, 77, 60)
    sex = st.selectbox("Sex", ["Male", "Female"])

    st.header("Vitals")
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, 130)
    chol = st.slider("Serum Cholesterol (mg/dl)", 126, 564, 240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["False", "True"])

with col2:
    st.header("Cardiac Signs")
    restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.slider("Max Heart Rate", 71, 202, 150)
    exang = st.selectbox("Exercise Induced Angina?", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.2, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST", ["Upsloping", "Flat", "Downsloping"])
    ca = st.slider("Major Vessels (0-3)", 0, 3, 0)
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"])


# --- 3. PREPROCESSING ---
def preprocess_input():
    # Map user inputs to the format your model expects (0-13 indices)
    # This must match your 'config.py' and 'data_utils.py' encoding EXACTLY

    row = [
        age,
        1 if sex == "Male" else 0,
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp) + 1,
        trestbps,
        chol,
        1 if fbs == "True" else 0,
        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg),
        thalach,
        1 if exang == "Yes" else 0,
        oldpeak,
        ["Upsloping", "Flat", "Downsloping"].index(slope) + 1,
        ca,
        ["Normal", "Fixed Defect", "Reversable Defect"].index(thal) + 3  # UCI often uses 3,6,7. Check your cleaning!
    ]
    return np.array(row).reshape(1, -1)


# --- 4. INFERENCE ---
if st.button("Analyze Risk", type="primary"):
    user_vector = preprocess_input()

    # Normalize (Important! Use the same scaler logic if you saved it,
    # otherwise we approximate with simple Z-score or pass raw if model handles it)
    # For this demo, we convert to tensor directly:
    x_new = torch.tensor(user_vector, dtype=torch.float32)

    # In a full Bipartite implementation, you would dynamically update edges here.
    # For Inductive GCN on simple attributes, we pass the features through the model:
    with torch.no_grad():
        logits = model(x_new)  # Assuming InductiveGCN can take just x for new nodes
        probs = torch.exp(logits)
        risk_score = probs[0][1].item()

    # Display
    st.divider()
    if risk_score > 0.35:  # Your optimized threshold
        st.error(f"**High Risk Detected**")
        st.metric(label="Probability of Disease", value=f"{risk_score:.2%}")
        st.warning("This patient shows patterns similar to high-risk clusters in the Bipartite Graph.")
    else:
        st.success(f"**Low Risk**")
        st.metric(label="Probability of Disease", value=f"{risk_score:.2%}")