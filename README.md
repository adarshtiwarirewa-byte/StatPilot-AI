# StatPilot AI 📊

**AI-Powered Statistical Analysis & Machine Learning Assistant**

StatPilot AI is a Streamlit web app that lets users upload a dataset and instantly explore it, query it in plain English, and train baseline machine learning models — all without writing a single line of code.

🔗 **Live App**: [https://statpilotai.streamlit.app]

---

## Features

### 1. Dataset Overview
Upload a CSV or Excel file and instantly see rows, columns, duplicate counts, data types, and missing value percentages in a clean summary table.

### 2. Exploratory Data Analysis (EDA)
- **Numeric columns**: Histogram, Box Plot, Violin Plot, or Scatter Plot (user-selectable) + correlation heatmap
- **Categorical columns**: Bar Chart, Pie Chart, or Frequency Table
- All visualizations powered by Plotly for interactivity

### 3. AI Chat With Dataset
Ask questions about your data in plain English. StatPilot AI uses the Groq LLM (Llama 3.3 70B) to convert your question into a SQL query, runs it against an in-memory SQLite representation of your dataset, and shows you both the result and the generated SQL — so you can learn and verify as you go.

### 4. Machine Learning
- Select a target column and features
- Auto-detects classification vs. regression (with manual override)
- Per-column missing-value handling (Auto/Median/Mode/Zero/Custom)
- Smart detection of numeric columns that are actually categorical
- Automatic encoding: One-Hot for low-cardinality, Target Encoding (with smoothing) for high-cardinality categorical features
- Trains and compares 2 baseline models per task (Logistic/Linear Regression + Random Forest)
- Optional sampling for datasets over 50,000 rows, to keep training fast

---

## Tech Stack

| Layer | Tools |
|---|---|
| Frontend/App | Streamlit |
| Data Handling | Pandas, NumPy |
| Visualization | Plotly |
| Machine Learning | scikit-learn |
| Natural Language → SQL | Groq API (Llama 3.3 70B Versatile) |
| Query Engine | SQLite (in-memory) |

---

## Project Structure

StatPilot-AI/
├── app.py # Main Streamlit app
├── modules/
│ ├── eda.py # EDA visualizations
│ └── ml_models.py # ML pipeline: preprocessing, training, evaluation
├── utils/
│ ├── data_loader.py # File upload & parsing
│ ├── sql_engine.py # NL-to-SQL execution engine
│ └── groq_client.py # Groq LLM API client
├── assets/
│ └── logo.png
├── .streamlit/
│ └── config.toml # Theme configuration
├── requirements.txt
└── README.md

---

## Running Locally

1. **Clone the repo**
```bash
   git clone https://github.com/adarshtiwarirewa-byte/StatPilot-AI.git
   cd StatPilot-AI
```

2. **Install dependencies**
```bash
   pip install -r requirements.txt
```

3. **Add your Groq API key**

   Create a `.env` file in the root directory:
   GROQ_API_KEY=your-api-key-here


   4. **Run the app**
```bash
   streamlit run app.py
```

---

## Deployment

This app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud). The Groq API key is managed via Streamlit's Secrets manager rather than a `.env` file in production.

---

## Roadmap

- [ ] RAG-based "Ask Your Dataset's Documentation" feature — upload a data dictionary PDF and query it with source-grounded answers
- [ ] Data Cleaning module (missing values, duplicates, outlier detection)
- [ ] Statistical Analysis module (hypothesis testing: t-test, ANOVA, Chi-Square)
- [ ] AutoML leaderboard with more model options
- [ ] SHAP-based model explainability
- [ ] Dark mode

---

## Author

**Adarsh Tiwari**
[GitHub](https://github.com/adarshtiwarirewa-byte)