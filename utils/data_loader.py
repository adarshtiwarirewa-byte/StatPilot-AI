import pandas as pd
import streamlit as st

def load_dataset(uploaded_file):
    """Reads an uploaded CSV or Excel file into a pandas DataFrame."""

    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".csv"):
            df =  pd.read_csv(uploaded_file)
        elif file_name.endswith((".xls", ".xlsx")):
            df =  pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        if df.empty : 
            st.warning("The uploaded file appears to be empty.")

        return df
    
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.exception(e)
        return None


def get_basic_info(df : pd.DataFrame) -> dict:
    """Returs basic structural info about the dataset."""

    return {
        "rows" : df.shape[0],
        "columns": df.shape[1],
        "column_names" : list(df.columns),
        "dtypes" : df.dtypes.astype(str).to_dict(),
        "missing_values" : df.isna().sum().to_dict(),
        "missing_percent" : (df.isna().sum()*100/len(df)).round(2).to_dict(),
        "duplicate_rows" : int(df.duplicated().sum())
    }