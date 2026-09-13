import pandas as pd
import streamlit as st 
import sqlite3
from utils.groq_client import generate_update_sql,is_query_safe



def provide_missing_values_section():
    df = st.session_state.df
    missing_summary = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isnull().sum().values,
        "Missing %": (df.isnull().sum().values / len(df) * 100).round(2)
    })
    missing_summary = missing_summary[missing_summary["Missing Count"] > 0].sort_values("Missing %", ascending=False)

    if missing_summary.empty:
        st.success("No missing values found.")
        return

    st.dataframe(missing_summary)

    # Collect all strategies for imputing each column or dropping rows
    strategies = {}
    custom_values = {}

    for col in missing_summary["Column"]:
        if pd.api.types.is_numeric_dtype(df[col]):
            options = ["Leave as is", "Drop rows", "Fill with Mean", "Fill with Median", "Fill with Custom Value"]
        else:
            options = ["Leave as is", "Drop rows", "Fill with Mode", "Fill with Custom Value"]

        strategy = st.selectbox(f"Strategy for '{col}'", options=options, key=f"clean_missing_{col}")
        strategies[col] = strategy

        if strategy == "Fill with Custom Value":
            if pd.api.types.is_numeric_dtype(df[col]):
                custom_values[col] = st.number_input(f"Custom value for '{col}'", key=f"custom_val_{col}")
            else:
                custom_values[col] = st.text_input(f"Custom value for '{col}'", key=f"custom_val_{col}")

    # ---- create a copy to work with (for safe side) and collect col names for dropping rows
    working_df = df.copy()
    rows_to_drop_cols = []

    # create copy of dataframe and clean it by the strategies
    for col, strategy in strategies.items():
        if strategy == "Drop rows":
            rows_to_drop_cols.append(col)
        elif strategy == "Fill with Mean":
            working_df[col] = working_df[col].fillna(working_df[col].mean())
        elif strategy == "Fill with Median":
            working_df[col] = working_df[col].fillna(working_df[col].median())
        elif strategy == "Fill with Mode":
            working_df[col] = working_df[col].fillna(working_df[col].mode()[0])
        elif strategy == "Fill with Custom Value":
            working_df[col] = working_df[col].fillna(custom_values[col])

    # drop cols at last to save from errors, and track before/after counts for safety check
    rows_before_drop = None
    rows_after_drop = None

    if rows_to_drop_cols:
        rows_before_drop = len(working_df)
        working_df = working_df.dropna(subset=rows_to_drop_cols)
        rows_after_drop = len(working_df)

    if st.button("Apply Missing Value Changes"):
        # Safety check 1: saari rows delete hone se roko (hard block)
        if rows_to_drop_cols and rows_after_drop == 0:
            st.error(
                f"⚠️ Applying 'Drop rows' on {rows_to_drop_cols} would remove ALL {rows_before_drop} rows "
                f"from your dataset (likely because these columns are entirely missing). "
                f"Operation cancelled — please choose a different strategy."
            )

        # Safety check 2: 50%+ rows delete hone par confirmation mango (soft warning)
        elif rows_to_drop_cols and rows_after_drop < rows_before_drop * 0.5:
            st.warning(
                f"⚠️ This will remove {rows_before_drop - rows_after_drop} out of {rows_before_drop} rows "
                f"({(1 - rows_after_drop / rows_before_drop) * 100:.1f}%)."
            )
            if st.checkbox("Yes, I understand and want to proceed", key="confirm_large_drop"):
                st.session_state.df = working_df
                st.success("Missing values handled!")
                st.rerun()

        # Normal case: koi risky drop nahi, seedha apply karo
        else:
            st.session_state.df = working_df
            st.success("Missing values handled!")
            st.rerun()


            
def provide_duplicate_section():
    df = st.session_state.df
    duplicate_count = df.duplicated().sum()
    
    if duplicate_count == 0:
        st.success("No duplicate rows found! ✅")
        return
    
    st.warning(f"Found {duplicate_count} duplicate rows.")
    
    if st.checkbox("Preview duplicate rows", key="preview_duplicates"):
        st.dataframe(df[df.duplicated(keep=False)])
    
    if st.button("Remove Duplicate Rows", key="remove_duplicates_btn"):
        cleaned_df = df.drop_duplicates(keep="first").reset_index(drop=True)
        st.session_state.df = cleaned_df
        st.success(f"Removed {duplicate_count} duplicate rows!")
        st.rerun()


def provide_outliers_section():
    df = st.session_state.df
    numeric_cols = df.select_dtypes(include = 'number').columns.tolist()

    if not numeric_cols : 
        st.info("No numeric cols found for outlier detection.")
        return

    selected_col = st.selectbox("Select a numeric column",numeric_cols,key="outlier_col_select")
    Q1 = df[selected_col].quantile(0.25)
    Q3 = df[selected_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    st.write(f"**Q1:** {Q1:.2f} | **Q3:** {Q3:.2f} | **IQR:** {IQR:.2f}")
    st.write(f"**Lower Bound:** {lower_bound:.2f} | **Upper Bound:** {upper_bound:.2f}")

    outlier_mask = (df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)
    outlier_count = outlier_mask.sum()
    
    if outlier_count == 0:
        st.success(f"No outliers found in '{selected_col}'! ✅")
    else:
        st.warning(f"Found {outlier_count} outliers in '{selected_col}'.")
        
        if st.checkbox("Preview outlier rows", key="preview_outliers"):
            st.dataframe(df[outlier_mask])


    action = st.radio("How do you want to handle these outliers ? ",
                      options = ["Do nothing","Remove Rows","Cap values to bounds"],
                      key = "outlier_action")
    if action == "Remove Rows" and st.button("Apply : Remove Outlier Rows",key = "remove_outliers_btn"):
        cleaned_df = df[~outlier_mask]
        st.session_state.df = cleaned_df
        st.success(f"Removed {outlier_count} outlier rows from the {selected_col}.")
        st.rerun()
    elif action == "Cap values to bounds" and st.button("Apply : Cap Outlier Values",key="cap_outlier_btn"):
        cleaned_df = df.copy()
        cleaned_df[selected_col] = cleaned_df[selected_col].clip(lower=lower_bound,upper=upper_bound)
        st.session_state.df = cleaned_df
        st.success(f"Capped {outlier_count} outlier rows from {selected_col}.")
        st.rerun()
    


def provide_dtype_conversion_section():
    
    df = st.session_state.df
    dtype_summary = pd.DataFrame({
        "Column": df.columns,
        "Current Type": df.dtypes.astype(str).values
    })
    st.dataframe(dtype_summary)
    
    selected_col = st.selectbox("Select column to convert", df.columns, key="dtype_col_select")
    target_type = st.selectbox(
        "Convert to",
        options=["Numeric (int/float)", "Text (string)", "Datetime", "Category"],
        key="dtype_target_select"
    )
    if st.button("Apply Conversion", key="apply_dtype_btn"):
        cleaned_df = df.copy()
        original_non_null = cleaned_df[selected_col].notna().sum()
        
        if target_type == "Numeric (int/float)":
            # Common cleanup before conversion: remove commas, currency symbols, whitespace
            cleaned_series = (
                cleaned_df[selected_col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(r"[₹$€£]", "", regex=True)
                .str.strip()
            )
            cleaned_df[selected_col] = pd.to_numeric(cleaned_series, errors="coerce")
        
        elif target_type == "Text (string)":
            cleaned_df[selected_col] = cleaned_df[selected_col].astype(str)
        
        elif target_type == "Datetime":
            cleaned_df[selected_col] = pd.to_datetime(cleaned_df[selected_col], errors="coerce")
        
        elif target_type == "Category":
            cleaned_df[selected_col] = cleaned_df[selected_col].astype("category")
        
        new_null_count = cleaned_df[selected_col].isna().sum()
        failed_conversions = new_null_count - (len(df) - original_non_null)
        
        if failed_conversions > 0:
            st.warning(f"{failed_conversions} values couldn't be converted and became NaN. Preview them below.")
            st.dataframe(df[cleaned_df[selected_col].isna() & df[selected_col].notna()])
        
        st.session_state.df = cleaned_df
        st.success(f"Column '{selected_col}' converted to {target_type}!")
        st.rerun()


def provide_custom_na_section():
    st.caption("If in your data  mean of missing is from a specific text/symbol (e.g.:- '?', '-', 'missing'), then specify here.")

    df = st.session_state.df
    custom_na_input = st.text_input(
        "Comma-separated placeholder values (case-sensitive)",
        placeholder="e.g. ?, missing, --, N.A.",
        key="custom_na_markers"
    )
    
    case_insensitive = st.checkbox("Case-insensitive match", key="case_insensitive_na")
    
    if st.button("Apply Custom NA Markers", key="apply_custom_na_btn"):
        if custom_na_input.strip():
            markers = [m.strip() for m in custom_na_input.split(",")]
            cleaned_df = df.copy()
            
            if case_insensitive:
                for col in cleaned_df.select_dtypes(include="object").columns:
                    cleaned_df[col] = cleaned_df[col].apply(
                        lambda x: pd.NA if isinstance(x, str) and x.strip().lower() in [m.lower() for m in markers] else x
                    )
            else:
                cleaned_df = cleaned_df.replace(markers, pd.NA)
            
            st.session_state.df = cleaned_df
            st.success(f"Replaced markers with NaN!")
            st.rerun()





def provide_ai_assisted_cleaning_section():
    df = st.session_state.df   # hamesha fresh state se lo, parameter se nahi
    
    
    st.caption(
        "Describe the change you want in plain English. The generated SQL will be shown "
        "for your review before anything is applied — nothing runs automatically."
    )
    
    user_instruction = st.text_input(
        "What would you like to change?",
        placeholder="e.g. set negative age values to NULL",
        key="ai_clean_instruction"
    )
    
    if st.button("Generate SQL", key="generate_clean_sql_btn"):
        if user_instruction.strip():
            with st.spinner("Generating SQL..."):
                generated_sql = generate_update_sql(user_instruction, "data", df)
            st.session_state.pending_sql = generated_sql
        else:
            st.warning("Please enter an instruction first.")
    
    # ---- Preview + Confirm/Cancel block ----
    if st.session_state.get("pending_sql"):
        st.markdown("**Generated SQL:**")
        st.code(st.session_state.pending_sql, language="sql")
        
        is_safe, blocked_reason = is_query_safe(st.session_state.pending_sql)
        
        if not is_safe:
            st.error(f"⚠️ This query was blocked for safety ({blocked_reason}). Try rephrasing your instruction.")
            if st.button("Dismiss", key="dismiss_unsafe_sql_btn"):
                st.session_state.pending_sql = None
                st.rerun()
        else:
            st.warning("Review the SQL above carefully. This will modify your working dataset.")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ Confirm & Apply", key="confirm_apply_sql_btn"):
                    apply_update_sql(st.session_state.pending_sql)
            
            with col2:
                if st.button("❌ Cancel", key="cancel_sql_btn"):
                    st.session_state.pending_sql = None
                    st.rerun()


def apply_update_sql(sql_query):
    df = st.session_state.df
    conn = sqlite3.connect(":memory:")
    
    try:
        df.to_sql("data", conn, index=False, if_exists="replace")
        cursor = conn.cursor()
        cursor.execute(sql_query)
        conn.commit()
        st.write(f"DEBUG: Rows affected = {cursor.rowcount}")   # temporary debug line
        
        updated_df = pd.read_sql("SELECT * FROM data", conn)
        st.session_state.df = updated_df
        st.session_state.pending_sql = None
        
        st.success("Changes applied successfully!")
        st.rerun()
    
    except Exception as e:
        st.error(f"Query failed: {e}")
    
    finally:
        conn.close()
            

def provide_data_cleaning_tab(df):
    tab1, tab2, tab3,tab4, tab5,tab6 = st.tabs(
                    ["Missing Value Markers","Dtypes Conversion","Missing Values", "Duplicates", "Outliers","AI-Assisted Cleaning"])

    with tab1:
        st.markdown("### Custom Missing Value Markers")
        provide_custom_na_section()
    with tab2:
        st.markdown("### Column Data Type Conversion") 
        provide_dtype_conversion_section()   
    with tab3:
        st.markdown("### Handle Missing Values")
        provide_missing_values_section()

    with tab4:
        st.markdown("### Handle Duplicate Values")
        provide_duplicate_section()
        
    with tab5:
        st.markdown("### Handle Outliers.")
        provide_outliers_section()
    with tab6:
        st.markdown("### AI-Assisted Cleaning")
        provide_ai_assisted_cleaning_section()


