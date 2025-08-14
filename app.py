# app.py
# Fortune 500 Data Scientist – Agentic AI Web App (Streamlit)
# -----------------------------------------------------------
# Goal: A single-file Streamlit app that orchestrates "agentic" building blocks
# to tackle common Fortune 500 data-science workflows end‑to‑end on the free tier.
#
# Key ideas
# - Runs fully locally on Streamlit Community Cloud (no paid APIs required).
# - Multi‑agent pattern implemented with small, composable Python classes.
# - Upload your dataset (CSV). The app does EDA → Feature Engineering → AutoML →
#   Explainability → Experimentation → Drift checks → Knowledge Graph →
#   (Optional) Web‑search snippets for nodes in the graph → (Optional) Quantum demo.
# - Uses widely available OSS libraries: pandas, numpy, scikit‑learn, matplotlib,
#   networkx, requests+bs4 (for optional web snippets), qiskit (quantum demo).
#
# Notes
# - No external LLM keys needed. The "agents" are rule‑based/pluggable so you can
#   later swap them for real LLMs if you have keys.
# - Web scraping can be flaky on any cloud; you can switch to "Local summaries".
# - Quantum section is an educational demo using qiskit's statevector simulator.
#
# -----------------------------------------------------------

from __future__ import annotations
import io
import os
import math
import textwrap
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, mean_squared_error,
    r2_score, precision_score, recall_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.inspection import permutation_importance

import networkx as nx

# Optional imports: search + parsing
try:
    import requests
    from bs4 import BeautifulSoup
    HAVE_WEB = True
except Exception:
    HAVE_WEB = False

# Optional imports: quantum demo
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    HAVE_QISKIT = True
except Exception:
    HAVE_QISKIT = False

st.set_page_config(page_title="Agentic AI Data Scientist", layout="wide")

# -----------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------

def safe_text(s: str, max_len: int = 280) -> str:
    s = " ".join(str(s).split())
    return (s[: max_len - 1] + "…") if len(s) > max_len else s

@st.cache_data(show_spinner=False)
def summarize_dataframe(df: pd.DataFrame) -> Dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    summary = {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing": int(df.isna().sum().sum()),
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "head": df.head(5).copy(),
    }
    return summary

@st.cache_data(show_spinner=False)
def corr_graph(df: pd.DataFrame, threshold: float = 0.3) -> nx.Graph:
    G = nx.Graph()
    for c in df.columns:
        G.add_node(c)
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.shape[1] >= 2:
        corr = num_df.corr(numeric_only=True)
        for i, c1 in enumerate(corr.columns):
            for j, c2 in enumerate(corr.columns):
                if j <= i:
                    continue
                w = corr.loc[c1, c2]
                if abs(w) >= threshold and not math.isnan(w):
                    G.add_edge(c1, c2, weight=float(w))
    return G

# -----------------------------------------------------------
# Agent primitives (no keys or paid APIs required)
# -----------------------------------------------------------

class BaseAgent:
    name = "agent"
    def run(self, *args, **kwargs):
        raise NotImplementedError

class PlannerAgent(BaseAgent):
    name = "Planner"
    def run(self, objective: str, kpis: List[str]) -> str:
        bullets = [
            "Clarify business objective and scope",
            "Audit data sources and access (governance, PII)",
            "Quick EDA to estimate signal and leakage",
            "Baseline model + backtesting",
            "Ship smallest viable decision tool",
            "Set up monitoring, drift, and retraining cadence",
        ]
        plan = f"Objective: {objective}\nSuccess metrics: {', '.join(kpis) if kpis else '—'}\n\nRoadmap:\n- " + "\n- ".join(bullets)
        return plan

class EDAAgent(BaseAgent):
    name = "EDA"
    def run(self, df: pd.DataFrame) -> Dict:
        s = summarize_dataframe(df)
        return s

class FeatureAgent(BaseAgent):
    name = "FeatureFactory"
    def run(self, df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, ColumnTransformer]:
        y = df[target]
        X = df.drop(columns=[target])
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ])
        pre = ColumnTransformer([
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ])
        return X, y, pre

class AutoMLAgent(BaseAgent):
    name = "AutoML"
    def run(self, X, y, pre: ColumnTransformer, task: str = "auto") -> Dict:
        # infer task
        if task == "auto":
            task = "classification" if y.nunique() <= max(10, int(0.01 * len(y))) else "regression"
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")

        if task == "classification":
            models = {
                "LogReg": LogisticRegression(max_iter=200, n_jobs=None),
                "RF": RandomForestClassifier(n_estimators=200, n_jobs=-1),
                "GBoost": GradientBoostingClassifier()
            }
            metric = "f1"
        else:
            models = {
                "LinReg": LinearRegression(),
                "RF": RandomForestRegressor(n_estimators=200, n_jobs=-1),
                "GBoost": GradientBoostingRegressor()
            }
            metric = "neg_root_mean_squared_error"

        results = {}
        best_name, best_score, best_pipe = None, -np.inf, None
        for name, model in models.items():
            pipe = Pipeline([("pre", pre), ("model", model)])
            try:
                scores = cross_val_score(pipe, X, y, cv=3, scoring=metric, n_jobs=-1)
                mean_score = float(np.mean(scores))
            except Exception as e:
                mean_score = float("nan")
            results[name] = mean_score
            if (not math.isnan(mean_score)) and mean_score > best_score:
                best_name, best_score, best_pipe = name, mean_score, pipe
        # Fit best
        if best_pipe is not None:
            best_pipe.fit(X, y)
        return {
            "task": task,
            "cv_metric": metric,
            "scores": results,
            "best_name": best_name,
            "best_cv": best_score,
            "best_pipeline": best_pipe
        }

class ExplainAgent(BaseAgent):
    name = "Explainer"
    def run(self, pipe: Pipeline, X: pd.DataFrame, y: pd.Series, task: str) -> Dict:
        # permutation importance on a sample for speed
        try:
            sample = X.sample(min(1000, len(X)), random_state=42)
            importances = permutation_importance(pipe, sample, y.loc[sample.index], n_repeats=5)
            names = pipe.named_steps["pre"].get_feature_names_out()
            idx = np.argsort(importances.importances_mean)[::-1]
            top = [(names[i], float(importances.importances_mean[i])) for i in idx[:20]]
        except Exception:
            top = []
        return {"top_importances": top}

class ExperimentAgent(BaseAgent):
    name = "Experimenter"
    def run(self, baseline: float, uplift: float, alpha: float = 0.05, power: float = 0.8) -> Dict:
        # Simple sample size calc for proportion metric (approximate)
        # baseline and uplift in decimals (e.g., 0.1 baseline CTR, 0.02 uplift)
        from statsmodels.stats.power import NormalIndPower
        effect = uplift
        analysis = NormalIndPower()
        n = analysis.solve_power(effect_size=effect / math.sqrt(baseline * (1 - baseline)),
                                 power=power, alpha=alpha, alternative='two-sided')
        return {"required_sample_per_arm": int(math.ceil(n))}

class DriftAgent(BaseAgent):
    name = "Monitor"
    def run(self, ref: pd.DataFrame, cur: pd.DataFrame) -> Dict:
        # Jensen‑Shannon divergence on numeric dists
        def js(p, q):
            p = p / (p.sum() + 1e-12)
            q = q / (q.sum() + 1e-12)
            m = 0.5 * (p + q)
            def kl(a, b):
                mask = (a > 0) & (b > 0)
                return np.sum(a[mask] * np.log(a[mask] / b[mask]))
            return 0.5 * kl(p, m) + 0.5 * kl(q, m)
        report = {}
        common_cols = [c for c in ref.columns if c in cur.columns]
        for c in common_cols:
            if pd.api.types.is_numeric_dtype(ref[c]):
                pr, bins = np.histogram(ref[c].dropna(), bins=30)
                pc, _ = np.histogram(cur[c].dropna(), bins=bins)
                report[c] = float(js(pr.astype(float), pc.astype(float)))
        return {"js_divergence": report}

class KnowledgeGraphAgent(BaseAgent):
    name = "KnowledgeGraph"
    def run(self, df: pd.DataFrame, threshold: float=0.3) -> Tuple[nx.Graph, Dict[str, str]]:
        G = corr_graph(df, threshold)
        # Local 1‑2 line notes for nodes
        notes = {}
        for node in G.nodes:
            role = "feature"
            if node.lower() in {"y", "label", "target"}:
                role = "target"
            deg = G.degree(node)
            notes[node] = safe_text(f"{node}: {role}; degree {deg}. Correlated with {deg} other fields above threshold.")
        return G, notes

class WebSnippetAgent(BaseAgent):
    name = "WebResearcher"
    def run(self, terms: List[str], per_term: int = 2) -> Dict[str, List[str]]:
        if not HAVE_WEB:
            return {t: ["Web libraries not available in this environment."] for t in terms}
        out = {}
        for t in terms:
            try:
                url = f"https://duckduckgo.com/html/?q={requests.utils.quote(t)}"
                r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                soup = BeautifulSoup(r.text, "html.parser")
                items = []
                for a in soup.select("a.result__a")[:per_term]:
                    title = a.get_text(" ")
                    items.append(safe_text(title))
                if not items:
                    items = ["No results parsed."]
                out[t] = items
            except Exception as e:
                out[t] = [f"Search failed: {e}"]
        return out

class QuantumAgent(BaseAgent):
    name = "QuantumDemo"
    def run(self, n_qubits: int = 2, reps: int = 1) -> Dict:
        if not HAVE_QISKIT:
            return {"status": "qiskit not installed"}
        qc = QuantumCircuit(n_qubits)
        # simple feature map style gates
        for r in range(reps):
            for i in range(n_qubits):
                qc.h(i)
                qc.rx(0.3 + 0.1*i, i)
            for i in range(n_qubits - 1):
                qc.cx(i, i+1)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities_dict()
        return {"circuit": qc.draw(output="text").single_string(), "probs": probs}

# -----------------------------------------------------------
# UI
# -----------------------------------------------------------

st.title("🧠 Agentic AI Data Scientist – Fortune 500 Workbench")
st.caption("EDA → AutoML → Explain → Experiment → Monitor → Knowledge Graph → Web Research → Quantum demo (no paid APIs)")

with st.sidebar:
    st.header("Project Setup")
    objective = st.text_input("Business Objective", value="Reduce churn by 10% in Q4")
    kpis = st.text_input("Success Metrics (comma‑sep)", value="F1, ROC AUC, Uplift")
    plan = PlannerAgent().run(objective, [k.strip() for k in kpis.split(",") if k.strip()])
    st.text_area("Plan", value=plan, height=160)

    st.markdown("---")
    st.subheader("Data Upload")
    uploaded = st.file_uploader("Upload CSV", type=["csv"]) 
    df: Optional[pd.DataFrame] = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding_errors="ignore")
    demo = st.checkbox("Load demo dataset (synthetic)")
    if demo:
        rng = np.random.default_rng(42)
        n = 800
        df = pd.DataFrame({
            "age": rng.integers(18, 70, n),
            "tenure_months": rng.integers(1, 120, n),
            "monthly_spend": np.round(rng.normal(60, 20, n).clip(5, 300), 2),
            "is_premium": rng.integers(0, 2, n),
            "city": rng.choice(["NY", "SF", "LA", "CHI", "DAL"], n),
        })
        # synthetic label
        logits = -3 + 0.03*df["age"] - 0.01*df["tenure_months"] + 0.02*df["monthly_spend"] + 0.6*df["is_premium"]
        p = 1/(1+np.exp(-logits))
        df["churned"] = (rng.random(n) < p).astype(int)

# Tabs
_tabs = [
    "Overview", "EDA", "Model Lab", "Explain", "Experiment", "Monitor", "Knowledge Graph", "Web Research", "Quantum"
]

if df is None:
    st.warning("Upload a CSV or enable the demo dataset from the sidebar.")
else:
    tab_over, tab_eda, tab_model, tab_explain, tab_exp, tab_monitor, tab_graph, tab_web, tab_quantum = st.tabs(_tabs)

    with tab_over:
        st.subheader("Dataset Snapshot")
        s = EDAAgent().run(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", s["rows"]) ; c2.metric("Columns", s["cols"]) ; c3.metric("Missing", s["missing"]) ; c4.metric("Numeric", len(s["numeric_cols"]))
        st.write("**Head**")
        st.dataframe(s["head"], use_container_width=True)
        st.write("**Numeric Columns**", s["numeric_cols"]) ; st.write("**Categorical Columns**", s["categorical_cols"]) 

    with tab_eda:
        st.subheader("Exploratory Analysis")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            col = st.selectbox("Plot numeric column", num_cols)
            fig = plt.figure()
            plt.hist(df[col].dropna(), bins=30)
            plt.title(f"Distribution of {col}")
            st.pyplot(fig)
            if len(num_cols) >= 2:
                col2 = st.selectbox("Correlation vs", [c for c in num_cols if c != col])
                fig2 = plt.figure()
                plt.scatter(df[col], df[col2], s=10)
                plt.title(f"Scatter: {col} vs {col2}")
                st.pyplot(fig2)
        else:
            st.info("No numeric columns to plot.")

    with tab_model:
        st.subheader("AutoML – Train baseline models")
        target = st.selectbox("Target column", [c for c in df.columns])
        if target:
            X, y, pre = FeatureAgent().run(df, target)
            task_choice = st.selectbox("Task", ["auto", "classification", "regression"], index=0)
            if st.button("Run AutoML"):
                with st.spinner("Training models (3‑fold CV)…"):
                    res = AutoMLAgent().run(X, y, pre, task_choice)
                st.success(f"Best: {res['best_name']} | CV ({res['cv_metric']}): {res['best_cv']:.4f}")
                st.json(res["scores"])
                st.session_state["best_pipe"] = res["best_pipeline"]
                st.session_state["task"] = res["task"]
                st.session_state["X"] = X
                st.session_state["y"] = y

    with tab_explain:
        st.subheader("Explain model (Permutation Importance)")
        if "best_pipe" in st.session_state:
            pipe = st.session_state["best_pipe"]
            X, y, task = st.session_state["X"], st.session_state["y"], st.session_state["task"]
            expl = ExplainAgent().run(pipe, X, y, task)
            top = expl.get("top_importances", [])
            if top:
                st.write("Top features by permutation importance:")
                imp_df = pd.DataFrame(top, columns=["feature", "importance"])            
                st.dataframe(imp_df, use_container_width=True)
                fig = plt.figure()
                plt.barh(imp_df["feature"], imp_df["importance"]) ; plt.title("Permutation importance")
                st.pyplot(fig)
            else:
                st.info("Could not compute permutation importances for this pipeline.")
        else:
            st.warning("Train a model first in the Model Lab tab.")

    with tab_exp:
        st.subheader("Experiment Design (A/B size)")
        baseline = st.number_input("Baseline rate (0‑1)", value=0.10, min_value=0.0001, max_value=0.9999)
        uplift = st.number_input("Minimum detectable uplift (absolute)", value=0.02, min_value=0.0001, max_value=0.5)
        alpha = st.number_input("Alpha", value=0.05, min_value=0.001, max_value=0.2)
        power = st.number_input("Power", value=0.8, min_value=0.5, max_value=0.99)
        if st.button("Compute sample size"):
            try:
                res = ExperimentAgent().run(baseline, uplift, alpha, power)
                st.success(f"Required sample per arm: {res['required_sample_per_arm']:,}")
            except Exception as e:
                st.error(f"Failed: {e}")

    with tab_monitor:
        st.subheader("Monitor drift between two datasets")
        st.write("Upload a **reference** and a **current** CSV with the same schema.")
        ref_up = st.file_uploader("Reference CSV", type=["csv"], key="ref")
        cur_up = st.file_uploader("Current CSV", type=["csv"], key="cur")
        if ref_up and cur_up:
            ref = pd.read_csv(ref_up)
            cur = pd.read_csv(cur_up)
            rep = DriftAgent().run(ref, cur)
            if rep["js_divergence"]:
                rep_df = pd.DataFrame(list(rep["js_divergence"].items()), columns=["column", "Jensen‑Shannon divergence"])
                st.dataframe(rep_df, use_container_width=True)
                fig = plt.figure()
                plt.bar(rep_df["column"], rep_df["Jensen‑Shannon divergence"]) ; plt.title("JS Divergence by column")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
            else:
                st.info("No numeric columns found to compare.")

    with tab_graph:
        st.subheader("Knowledge Graph of Correlations")
        threshold = st.slider("Correlation threshold (abs)", 0.1, 0.9, 0.3, 0.05)
        G, local_notes = KnowledgeGraphAgent().run(df, threshold)
        st.write(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
        # Simple spring layout plot
        pos = nx.spring_layout(G, seed=42, k=0.5)
        fig = plt.figure(figsize=(6, 5))
        nx.draw(G, pos, with_labels=True, node_size=400, font_size=8)
        st.pyplot(fig)
        st.write("Local 1‑2 line notes:")
        for k, v in local_notes.items():
            st.write(f"- **{k}**: {v}")
        st.session_state["graph_nodes"] = list(G.nodes)

    with tab_web:
        st.subheader("Agentic Web Research for Graph Terms (Optional)")
        if "graph_nodes" not in st.session_state or not st.session_state["graph_nodes"]:
            st.info("Build the Knowledge Graph first to populate terms.")
        else:
            mode = st.radio("Mode", ["Local summaries", "Web snippets (experimental)"])
            terms = st.multiselect("Select terms to research", st.session_state["graph_nodes"], default=st.session_state["graph_nodes"][:10])
            if st.button("Run agents"):
                if mode == "Local summaries":
                    st.success("Generated 1‑2 line local summaries for selected terms.")
                    st.write({t: f"{t}: feature in dataset; interactions visible in correlation graph." for t in terms})
                else:
                    res = WebSnippetAgent().run(terms, per_term=2)
                    st.write(res)

    with tab_quantum:
        st.subheader("Quantum Computing Demo (Qiskit)")
        if not HAVE_QISKIT:
            st.warning("Qiskit not installed. Add 'qiskit' to requirements.txt to enable.")
        else:
            n_qubits = st.slider("Qubits", 2, 5, 2)
            reps = st.slider("Layers", 1, 3, 1)
            if st.button("Run quantum demo"):
                res = QuantumAgent().run(n_qubits, reps)
                st.code(res.get("circuit", ""))
                probs = res.get("probs", {})
                if probs:
                    prob_df = pd.DataFrame([{"state": k, "prob": v} for k, v in probs.items()])
                    st.dataframe(prob_df, use_container_width=True)
                    fig = plt.figure()
                    plt.bar(prob_df["state"], prob_df["prob"]) ; plt.title("State probabilities")
                    st.pyplot(fig)

st.markdown("""
---
**Ethics & Limits**: This tool accelerates and automates a large fraction of a data scientist's workflow, but it does not replace human oversight, domain judgment, or accountability. Always validate results with stakeholders, ensure compliance (GDPR/CCPA/ISO), and test in controlled rollouts.
""")
