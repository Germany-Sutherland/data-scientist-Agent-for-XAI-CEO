# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from statsmodels.stats.power import NormalIndPower

# Optional Qiskit
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    HAVE_QISKIT = True
except Exception:
    HAVE_QISKIT = False

# ------------------------------
# Streamlit UI Setup
# ------------------------------
st.set_page_config(page_title="Fortune 500 AI Data Scientist", layout="wide")
st.title("🤖 Agentic AI Data Scientist for Fortune 500")

# ------------------------------
# File Upload
# ------------------------------
uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
if uploaded:
    if uploaded.size > 50 * 1024 * 1024:
        st.error("❌ File too large (>50 MB). Please upload a smaller dataset.")
        st.stop()
    df = pd.read_csv(uploaded)
    st.success(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
else:
    st.info("Upload a CSV file to start.")
    df = None

# ------------------------------
# EDA Agent
# ------------------------------
def run_eda(data):
    st.subheader("📊 Exploratory Data Analysis")
    st.write("First 5 rows:")
    st.dataframe(data.head())
    st.write("Summary statistics:")
    st.write(data.describe())

    # Correlation heatmap
    corr = data.corr(numeric_only=True)
    fig, ax = plt.subplots()
    cax = ax.matshow(corr, cmap="coolwarm")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    fig.colorbar(cax)
    st.pyplot(fig)

# ------------------------------
# AutoML Agent
# ------------------------------
def run_automl(data, target):
    st.subheader("🤖 AutoML Agent")
    if target not in data.columns:
        st.error(f"Target column '{target}' not found.")
        return

    X = data.drop(columns=[target])
    y = data[target]

    if y.dtype == "object" or len(y.unique()) < 10:
        task = "classification"
        models = [
            ("LogisticRegression", LogisticRegression(max_iter=500)),
            ("RandomForestClassifier", RandomForestClassifier(n_estimators=100))
        ]
    else:
        task = "regression"
        models = [
            ("LinearRegression", LinearRegression()),
            ("RandomForestRegressor", RandomForestRegressor(n_estimators=100))
        ]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model, best_score = None, -np.inf
    for name, model in models:
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        try:
            scores = cross_val_score(pipe, X_train, y_train, cv=3)
            avg_score = np.mean(scores)
        except Exception:
            avg_score = -np.inf

        if avg_score > best_score:
            best_score, best_model = avg_score, pipe

    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)

    if task == "classification":
        score = accuracy_score(y_test, preds)
        st.write(f"Best Model Accuracy: {score:.3f}")
    else:
        score = mean_squared_error(y_test, preds, squared=False)
        st.write(f"Best Model RMSE: {score:.3f}")

# ------------------------------
# Knowledge Graph Agent
# ------------------------------
def run_knowledge_graph(data):
    st.subheader("🧠 Knowledge Graph")
    G = nx.Graph()
    for col in data.columns:
        G.add_node(col)
    for i, col1 in enumerate(data.columns):
        for col2 in data.columns[i+1:]:
            G.add_edge(col1, col2)
    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G)
        plt.figure()
        nx.draw(G, pos, with_labels=True, node_size=400, font_size=8)
        st.pyplot(plt.gcf())
    else:
        st.write("No nodes to display.")

# ------------------------------
# Web Research Agent
# ------------------------------
def run_web_agent(terms):
    st.subheader("🌐 Web Research Agent")
    results = {}
    for term in terms:
        try:
            r = requests.get(f"https://duckduckgo.com/html/?q={term}", timeout=5)
            soup = BeautifulSoup(r.text, "html.parser")
            snippet = soup.find("a", {"class": "result__snippet"})
            if snippet:
                results[term] = snippet.get_text()
            else:
                results[term] = "No snippet found."
        except Exception:
            results[term] = "Web search unavailable."
    st.write(results)

# ------------------------------
# Quantum Agent (optional)
# ------------------------------
def run_quantum_demo():
    st.subheader("⚛️ Quantum Computing Demo")
    if not HAVE_QISKIT:
        st.warning("Qiskit not installed — quantum demo unavailable.")
        return
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    state = Statevector.from_instruction(qc)
    st.write("Quantum Statevector:", state)

# ------------------------------
# Tabs Layout
# ------------------------------
if df is not None:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["EDA", "AutoML", "Knowledge Graph", "Web Agent", "Quantum"]
    )

    with tab1:
        run_eda(df)

    with tab2:
        target_col = st.selectbox("Select target column", df.columns)
        if st.button("Run AutoML"):
            run_automl(df, target_col)

    with tab3:
        run_knowledge_graph(df)

    with tab4:
        terms = st.multiselect("Select terms for web research", df.columns)
        if st.button("Run Web Agent"):
            run_web_agent(terms)

    with tab5:
        run_quantum_demo()
