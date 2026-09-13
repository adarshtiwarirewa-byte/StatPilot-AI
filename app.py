import streamlit as st
import pandas as pd

from utils.data_loader import load_dataset, get_basic_info
from utils.sql_engine import create_sql_table,run_sql_query
from utils.groq_client import generate_sql_query
from modules.ml_models import detect_problem_type, preprocess_data, train_and_evaluate,detect_possibly_categorical_numeric
from modules.eda import (
    get_numeric_columns,
    plot_histogram,
    plot_boxplot,
    plot_violin,
    plot_scatter,
    plot_correlation_heatmap,
    get_categorical_columns,
    plot_bar_chart,
    plot_pie_chart,
    get_frequency_table,
)

from modules.rag_QA import answer_question
from utils.pdf_parser import extract_text_from_pdf
from utils.embedding_engine import chunk_text, embed_chunks
from utils.vector_store import VectorStore
from modules.data_cleaning  import  provide_data_cleaning_tab, provide_missing_values_section, provide_outliers_section,provide_duplicate_section
from modules.statistical_analysis import render_statistical_analysis_tab



st.set_page_config(page_title="StatPilot AI", page_icon="assets/logo_transparent_v2.png", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Overall padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 12px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        font-weight: 500;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        border: 1px solid #E2E8F0;
    }
    .stButton > button:hover {
        border-color: #2563EB;
        color: #2563EB;
    }

    /* Dataframe corners */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
    }

    /* Sidebar - dark override */
    section[data-testid="stSidebar"] {
        background-color: #0E2140;
        border-right: 1px solid #1E3A5F;
    }

    section[data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }

    section[data-testid="stSidebar"] .stCaption, 
    section[data-testid="stSidebar"] small {
        color: #94A3B8 !important;
    }

    /* Sidebar file uploader box */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background-color: #16305A;
        border: 1px dashed #2A4A75;
        border-radius: 8px;
    }

    /* Sidebar divider */
    section[data-testid="stSidebar"] hr {
        border-color: #1E3A5F;
    }

    /* Headers */
    h1, h2, h3 {
        font-weight: 600;
        color: #0F172A;
    }

    /* Code blocks (SQL query display) */
    .stCodeBlock {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    with st.sidebar:
     st.image("assets/logo_transparent_v2.png", width='stretch')
     st.markdown(
          "<p style='text-align: center; color: #94A3B8; font-size: 16px; margin-top: -10px;'>"
          "AI-Powered Statistical Analysis Assistant</p>",
          unsafe_allow_html=True,
     )
    

     uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

    st.divider()
    st.markdown(
        """
        <div style='font-size: 13px; color: gray;'>
        Built with ❤️ using Streamlit, Pandas, scikit-learn & Groq<br>
        <a href='https://github.com/adarshtiwarirewa-byte/StatPilot-AI' target='_blank'>GitHub Repo</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- MAIN AREA ----------------
if uploaded_file is None:
    st.markdown("## Welcome to StatPilot AI 👋")
    st.write(
        "Upload your dataset from the sidebar and get instant dataset overview, "
        "exploratory analysis, AI-powered Q&A, and machine learning — all in one place."
    )
    st.info("👈 Upload a CSV or Excel file from the sidebar to get started.")
else:
    df_uploaded = load_dataset(uploaded_file)

    # Storing original and copying df ,  We will work on df like cleaning and all but not in original_df. 
    # Due to this we can return to our original dataframe whenever we wanted.

    if df_uploaded is not None:
        if "current_file_id" not in st.session_state or st.session_state.current_file_id != uploaded_file.file_id:
            st.session_state.df_original = df_uploaded.copy()
            st.session_state.df = df_uploaded.copy()
            st.session_state.current_file_id = uploaded_file.file_id

        if "pending_sql" not in st.session_state:
            st.session_state.pending_sql = None

        st.success("Dataset loaded successfully!")
        df = st.session_state.df
        tab1, tab2, tab3, tab4, tab5 ,tab6,tab7= st.tabs([
                                                    "📁 Overview",
                                                    "Know Documentaion of Dataset(RAG)",
                                                    "📈 EDA",
                                                    "🤖 AI Chat(SQL)",
                                                    "Data Cleaning",
                                                    "🧠 Machine Learning",
                                                    "📊 Statistical Analysis"
                                                ])

        # ---------------- TAB 1: OVERVIEW ----------------
        with tab1:
            st.caption("Get a quick summary of your dataset's structure, size, and quality.")
            info = get_basic_info(df)

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", info["rows"])
            col2.metric("Columns", info["columns"])
            col3.metric("Duplicate Rows", info["duplicate_rows"])

            st.subheader("Preview")
            st.dataframe(df.sample(10), use_container_width=True)

            st.subheader("Column Info")
            col_info_df = pd.DataFrame({
                "Column": info["dtypes"].keys(),
                "Data Type": info["dtypes"].values(),
                "Missing Values": info["missing_values"].values(),
                "Missing %": info["missing_percent"].values(),
            })
            st.dataframe(col_info_df, use_container_width=True)


        # --------------------TAB 2 : Ask Documentation of the DATASET---------------
        with tab2:
            st.subheader("Ask Your Dataset's Documentation")
            st.caption("Upload a PDF (data dictionary, README, or documentation) and ask questions about it.")
            
            uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], key="rag_pdf_uploader")

            MAX_FILE_SIZE_MB = 10
            MAX_QUESTIONS_PER_SESSION = 15
            
            if uploaded_pdf is not None:

                file_size_mb = uploaded_pdf.size / (1024 * 1024)
    
                if file_size_mb > MAX_FILE_SIZE_MB:
                    st.error(f"File too large ({file_size_mb:.1f}MB). Please upload a PDF under {MAX_FILE_SIZE_MB}MB.")
                    st.stop()


                # Process only once per uploaded file (avoid re-processing on every rerun)


                if "rag_vector_store" not in st.session_state or st.session_state.get("rag_pdf_name") != uploaded_pdf.name:
                    with st.spinner("Processing document... (extracting text, creating embeddings)"):
                        pages = extract_text_from_pdf(uploaded_pdf)
                        chunks = chunk_text(pages)
                        chunks_with_embeddings = embed_chunks(chunks)
                        
                        store = VectorStore(embedding_dim=384)
                        store.add_chunks(chunks_with_embeddings)
                        
                        st.session_state["rag_vector_store"] = store
                        st.session_state["rag_pdf_name"] = uploaded_pdf.name
                        st.session_state["rag_chat_history"] = []
                        st.session_state['rag_question_count'] = 0
                    st.success(f"Document processed! ({len(chunks)} chunks created)")
                
                st.divider()
                
                # Display chat history
                for msg in st.session_state.get("rag_chat_history", []):
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])
                        if msg["role"] == "assistant" and msg.get("sources"):
                            st.caption(f"Source: Page(s) {', '.join(map(str, msg['sources']))}")

                st.caption(f"Questions used: {st.session_state.get('rag_question_count', 0)}/{MAX_QUESTIONS_PER_SESSION}")

                # Chat input with session limit
                if st.session_state.get("rag_question_count", 0) >= MAX_QUESTIONS_PER_SESSION:
                    st.warning(f"You've reached the limit of {MAX_QUESTIONS_PER_SESSION} questions for this session. Please refresh the page to continue.")
                else:
                    user_question = st.chat_input("Ask a question about the document...")

                    if user_question:
                        st.session_state["rag_chat_history"].append({"role": "user", "content": user_question})
                        st.session_state["rag_question_count"] += 1

                        with st.spinner("Thinking..."):
                            result = answer_question(user_question, st.session_state["rag_vector_store"],top_k=8)

                        st.session_state["rag_chat_history"].append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"]
                        })
                        st.rerun()
            else:
                st.info("Upload a PDF to get started.")

        # ---------------- TAB 3: EDA ----------------
        with tab3:
            st.caption("Visualize distributions and relationships hidden in your data.")

            numeric_cols = get_numeric_columns(df)
            categorical_cols = get_categorical_columns(df)

            # ---------- Numeric EDA ----------
            st.subheader("📊 Numeric Column Analysis")

            if len(numeric_cols) == 0:
                st.info("No numeric columns found.")
            else:
                chart_type = st.selectbox(
                    "Chart type",
                    ["Histogram", "Box Plot", "Violin Plot", "Scatter Plot"],
                )

                if chart_type == "Scatter Plot":
                    if len(numeric_cols) < 2:
                        st.info("Need at least 2 numeric columns for a scatter plot.")
                    else:
                        col_a, col_b = st.columns(2)
                        x_col = col_a.selectbox("X-axis", numeric_cols, index=0)
                        y_col = col_b.selectbox("Y-axis", numeric_cols, index=1)
                        fig = plot_scatter(df, x_col, y_col)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    selected_col = st.selectbox("Choose a numeric column", numeric_cols)

                    if chart_type == "Histogram":
                        fig = plot_histogram(df, selected_col)
                    elif chart_type == "Box Plot":
                        fig = plot_boxplot(df, selected_col)
                    else:
                        fig = plot_violin(df, selected_col)

                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("Correlation Heatmap")
                heatmap_fig = plot_correlation_heatmap(df, numeric_cols)
                if heatmap_fig is not None:
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                else:
                    st.info("Need at least 2 numeric columns for a correlation heatmap.")

            st.divider()

            # ---------- Categorical EDA ----------
            st.subheader("🗂️ Categorical Column Analysis")

            if len(categorical_cols) == 0:
                st.info("No categorical columns found.")
            else:
                cat_chart_type = st.selectbox(
                    "Chart type",
                    ["Bar Chart", "Pie Chart", "Frequency Table"],
                    key="cat_chart_type",
                )
                selected_cat_col = st.selectbox("Choose a categorical column", categorical_cols)

                if cat_chart_type == "Bar Chart":
                    fig = plot_bar_chart(df, selected_cat_col)
                    st.plotly_chart(fig, use_container_width=True)
                elif cat_chart_type == "Pie Chart":
                    fig = plot_pie_chart(df, selected_cat_col)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    freq_df = get_frequency_table(df, selected_cat_col)
                    st.dataframe(freq_df, use_container_width=True)

        # ---------------- TAB 4: AI CHAT ----------------
        with tab4:
            conn = create_sql_table(df, table_name="dataset")
            st.caption("Ask questions about your data in plain English — AI converts it into SQL automatically.")

            user_question = st.text_input("Ask a question about your data (in plain English)")

            if st.button("Ask"):
                if user_question.strip() == "":
                    st.warning("Please type a question first.")
                else:
                    with st.spinner("Thinking..."):
                        sql_query = generate_sql_query(
                         question=user_question,
                         table_name="dataset",df=df)

                    st.code(sql_query, language="sql")

                    result_df = run_sql_query(conn, sql_query)
                    st.dataframe(result_df, use_container_width=True)


        # ----------------TAB 5 : DATA CLEANING -------------------
        with tab5:
            if st.button("🔄 Reset to Original Dataset"):
                st.session_state.df = st.session_state.df_original.copy()
                st.rerun()
            provide_data_cleaning_tab(df)



        # ---------------- TAB 5: MACHINE LEARNING ----------------
        with tab6:
            st.caption("Select a target column and compare baseline model performance instantly.")
            target_column = st.selectbox(
                "Select the target column (what you want to predict)",
                df.columns,
            )
            
            available_features = [col for col in df.columns if col != target_column]
            feature_columns = st.multiselect("Select feature columns to use for training models",
                                             options= available_features,default=available_features)
            
            

            if len(df) > 50000:
                use_sample = st.checkbox(
                    f"Dataset has {len(df):,} rows — train on a random sample of 50,000 rows for faster results?",
                    value=True,
                )
            else:
                use_sample = False


            categorical_threshold = st.slider(
                                                  "Flag numeric columns with up to this many unique values as 'possibly categorical'",
                                                  min_value=2,
                                                  max_value=50,
                                                  value=10,
                                             )
            possibly_categorical = detect_possibly_categorical_numeric(df, feature_columns)
          
            categorical_override = []
            if possibly_categorical:
                st.write("**These numeric columns have few unique values — are any of them actually categories?**")
                categorical_override = st.multiselect(
                    "Select columns to treat as categorical (not continuous numeric)",
                    possibly_categorical,
                )
            

            auto_detected_type = detect_problem_type(df, target_column)

            problem_type = st.radio(
                    f"Problem type (auto-detected: {auto_detected_type.capitalize()})",
                    ["classification", "regression"],
                    index=0 if auto_detected_type == "classification" else 1,
                    horizontal=True,
                    )
            
            if st.button("Train Models"):
                if len(feature_columns) == 0 :
                    st.warning("Please select at least one feature column.")
                else:
                    with st.spinner("Training models..."):
                         training_df = df.sample(n=50000, random_state=42) if use_sample else df
                         X, y = preprocess_data(training_df,target_column,feature_columns)

                         if X.shape[1] == 0:
                              st.warning("No feature columns available after removing the target column.")
                         else:
                              results_df = train_and_evaluate(X, y, problem_type)
                              st.subheader("Model Results")
                              st.dataframe(results_df, use_container_width=True)



        #----------------------Tab7 : Statistical Analysis------------------

        with tab7 :
            render_statistical_analysis_tab()