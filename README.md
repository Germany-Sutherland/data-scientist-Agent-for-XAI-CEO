# data-scientist-Agent-for-XAI-CEO

# Fortune 500 Agentic AI Data Scientist

A free-tier-friendly Streamlit app that simulates the workflow of a Fortune 500 data scientist using:
- Agentic AI logic (rule-based agents)
- EDA automation
- Lightweight AutoML
- Explainability
- Knowledge Graph building
- Optional Web Research
- Optional Quantum Computing demo

## 🚀 Deployment

1. Fork this repo or download it.
2. Push to your own **public GitHub repo**.
3. Go to [Streamlit Cloud](https://share.streamlit.io/), click **"New app"**.
4. Connect your GitHub repo, set `Main file path` to `app.py`.
5. Deploy — done!

## 💡 Notes
- Works on free Streamlit Community Cloud (memory limit ~1GB).
- Avoid large datasets (>50 MB).
- Quantum tab requires `qiskit`. If not needed, remove from `requirements.txt` and `app.py`.

## 📂 Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
