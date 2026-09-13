import pandas as pd  
import streamlit as st 
from scipy import stats


def render_statistical_analysis_tab():
    df = st.session_state.df
    st.subheader("📊 Statistical Analysis")
    
    test_type = st.selectbox(
        "Select a statistical test",
        options=[
            "Normality Test (Shapiro-Wilk)",
            "T-Test (Compare 2 Groups)",
            "ANOVA (Compare 3+ Groups)",
            "Chi-Square Test (Categorical Association)",
            "Correlation Test (Numeric Relationship)"
        ],
        key="stat_test_select"
    )
    
    st.divider()
    
    if test_type == "Normality Test (Shapiro-Wilk)":
        render_normality_test(df)
    elif test_type == "T-Test (Compare 2 Groups)":
        render_ttest(df)
    elif test_type == "ANOVA (Compare 3+ Groups)":
        render_anova(df)
    elif test_type == "Chi-Square Test (Categorical Association)":
        render_chi_square(df)
    elif test_type == "Correlation Test (Numeric Relationship)":
        render_correlation(df)



def render_normality_test(df):
    st.markdown("### Normality Test (Shapiro-Wilk)")
    st.caption("Checks whether a numeric column follows a normal distribution.")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    
    if not numeric_cols:
        st.info("No numeric columns available for this test.")
        return
    
    selected_col = st.selectbox("Select a numeric column", numeric_cols, key="normality_col_select")
    
    data = df[selected_col].dropna()
    
    if len(data) < 3:
        st.warning("Not enough non-missing values to run this test (minimum 3 required).")
        return

        if len(data) > 5000:
            st.info("ℹ️ Note: Shapiro-Wilk test can be overly sensitive with large samples (n > 5000) — even minor deviations may appear statistically significant.")
    
    if st.button("Run Normality Test", key="run_normality_btn"):
        stat, p_value = stats.shapiro(data)
        
        st.markdown("#### Results")
        st.write(f"**Test Statistic:** {stat:.4f}")
        st.write(f"**P-value:** {p_value:.4f}")
        
        alpha = 0.05
        if p_value < alpha:
            st.error(f"**Decision:** Reject H0 (p = {p_value:.4f} < {alpha})")
            st.write(f"**Interpretation:** '{selected_col}' does **not** appear to follow a normal distribution.")
        else:
            st.success(f"**Decision:** Fail to reject H0 (p = {p_value:.4f} ≥ {alpha})")
            st.write(f"**Interpretation:** '{selected_col}' appears to follow a normal distribution.")




def render_ttest(df):
    st.markdown("### T-Test (Compare 2 Groups)")
    st.caption("Compares the mean of a numeric column across exactly 2 groups.")
    
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    
    # Sirf woh categorical columns jinme exactly 2 unique values hon
    valid_group_cols = [
        col for col in categorical_cols 
        if df[col].nunique() == 2
    ]
    
    if not numeric_cols or not valid_group_cols:
        st.info("Need at least one numeric column and one categorical column with exactly 2 groups.")
        return
    
    numeric_col = st.selectbox("Select numeric column", numeric_cols, key="ttest_numeric_select")
    group_col = st.selectbox("Select grouping column (2 groups)", valid_group_cols, key="ttest_group_select")
    
    groups = df[group_col].dropna().unique()
    group1_data = df[df[group_col] == groups[0]][numeric_col].dropna()
    group2_data = df[df[group_col] == groups[1]][numeric_col].dropna()
    
    st.write(f"**Group '{groups[0]}':** {len(group1_data)} samples | **Group '{groups[1]}':** {len(group2_data)} samples")
    
    if len(group1_data) < 2 or len(group2_data) < 2:
        st.warning("Each group needs at least 2 non-missing values to run this test.")
        return
    
    if st.button("Run T-Test", key="run_ttest_btn"):
        stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
        
        st.markdown("#### Results")
        st.write(f"**T-statistic:** {stat:.4f}")
        st.write(f"**P-value:** {p_value:.4f}")
        st.write(f"**Mean ({groups[0]}):** {group1_data.mean():.2f} | **Mean ({groups[1]}):** {group2_data.mean():.2f}")
        
        alpha = 0.05
        if p_value < alpha:
            st.error(f"**Decision:** Reject H0 (p = {p_value:.4f} < {alpha})")
            st.write(f"**Interpretation:** There **is** a statistically significant difference in '{numeric_col}' between '{groups[0]}' and '{groups[1]}'.")
        else:
            st.success(f"**Decision:** Fail to reject H0 (p = {p_value:.4f} ≥ {alpha})")
            st.write(f"**Interpretation:** There is **no** statistically significant difference in '{numeric_col}' between '{groups[0]}' and '{groups[1]}'.")



def render_anova(df):
    st.markdown("### ANOVA (Compare 3+ Groups)")
    st.caption("Compares the mean of a numeric column across 3 or more groups.")
    
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    
    # Sirf woh categorical columns jinme 3+ unique values hon (aur bahut zyada bhi na hon)
    valid_group_cols = [
        col for col in categorical_cols 
        if df[col].nunique() >= 3 and df[col].nunique() <= 20
    ]
    
    if not numeric_cols or not valid_group_cols:
        st.info("Need at least one numeric column and one categorical column with 3-20 groups.")
        return
    
    numeric_col = st.selectbox("Select numeric column", numeric_cols, key="anova_numeric_select")
    group_col = st.selectbox("Select grouping column (3+ groups)", valid_group_cols, key="anova_group_select")
    
    groups = df[group_col].dropna().unique()
    group_data = [df[df[group_col] == g][numeric_col].dropna() for g in groups]
    
    # Har group mein minimum 2 values honi chahiye
    valid_groups = [(g, d) for g, d in zip(groups, group_data) if len(d) >= 2]
    
    if len(valid_groups) < 3:
        st.warning("At least 3 groups need 2+ non-missing values each to run ANOVA.")
        return
    
    group_summary = pd.DataFrame({
        "Group": [g for g, d in valid_groups],
        "Count": [len(d) for g, d in valid_groups],
        "Mean": [d.mean() for g, d in valid_groups]
    })
    st.dataframe(group_summary)
    
    if st.button("Run ANOVA", key="run_anova_btn"):
        stat, p_value = stats.f_oneway(*[d for g, d in valid_groups])
        
        st.markdown("#### Results")
        st.write(f"**F-statistic:** {stat:.4f}")
        st.write(f"**P-value:** {p_value:.4f}")
        
        alpha = 0.05
        if p_value < alpha:
            st.error(f"**Decision:** Reject H0 (p = {p_value:.4f} < {alpha})")
            st.write(f"**Interpretation:** At least one group's mean of '{numeric_col}' is significantly different across '{group_col}' categories.")
            st.caption("ℹ️ ANOVA tells us a difference exists somewhere, but not which specific groups differ. A post-hoc test (e.g. Tukey's HSD) would be needed for pairwise comparisons.")

        else:
            st.success(f"**Decision:** Fail to reject H0 (p = {p_value:.4f} ≥ {alpha})")
            st.write(f"**Interpretation:** No statistically significant difference found in '{numeric_col}' across '{group_col}' categories.")




def render_chi_square(df):
    st.markdown("### Chi-Square Test (Categorical Association)")
    st.caption("Checks whether two categorical variables are statistically related.")
    
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    
    # Bahut high-cardinality columns ko exclude karo (jaise IDs)
    valid_cols = [col for col in categorical_cols if 2 <= df[col].nunique() <= 20]
    
    if len(valid_cols) < 2:
        st.info("Need at least two categorical columns (with 2-20 unique categories each) for this test.")
        return
    
    col1 = st.selectbox("Select first categorical column", valid_cols, key="chi_col1_select")
    col2 = st.selectbox("Select second categorical column", 
                         [c for c in valid_cols if c != col1], 
                         key="chi_col2_select")
    
    contingency_table = pd.crosstab(df[col1], df[col2])
    
    st.markdown("#### Contingency Table (Observed Frequencies)")
    st.dataframe(contingency_table)
    
    if st.button("Run Chi-Square Test", key="run_chi_square_btn"):
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Sanity check: expected frequencies bahut chhoti toh test unreliable ho sakta hai
        min_expected = expected.min()
        
        st.markdown("#### Results")
        st.write(f"**Chi-Square Statistic:** {chi2_stat:.4f}")
        st.write(f"**Degrees of Freedom:** {dof}")
        st.write(f"**P-value:** {p_value:.4f}")
        
        if min_expected < 5:
            st.warning(
                f"⚠️ Some expected cell frequencies are below 5 (minimum: {min_expected:.2f}). "
                f"Chi-Square results may be unreliable — consider combining categories or using Fisher's Exact Test instead."
            )
        
        alpha = 0.05
        if p_value < alpha:
            st.error(f"**Decision:** Reject H0 (p = {p_value:.4f} < {alpha})")
            st.write(f"**Interpretation:** There **is** a statistically significant association between '{col1}' and '{col2}'.")
        else:
            st.success(f"**Decision:** Fail to reject H0 (p = {p_value:.4f} ≥ {alpha})")
            st.write(f"**Interpretation:** There is **no** statistically significant association between '{col1}' and '{col2}' — they appear independent.")




def render_correlation(df):
    st.markdown("### Correlation Test")
    st.caption("Checks the strength and significance of a relationship between two numeric variables.")
    
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    
    if len(numeric_cols) < 2:
        st.info("Need at least two numeric columns for this test.")
        return
    
    col1 = st.selectbox("Select first numeric column", numeric_cols, key="corr_col1_select")
    col2 = st.selectbox("Select second numeric column", 
                         [c for c in numeric_cols if c != col1], 
                         key="corr_col2_select")
    
    method = st.radio(
        "Select correlation method",
        options=["Pearson (linear, assumes normal distribution)", "Spearman (rank-based, no normality assumption)"],
        key="corr_method_select"
    )
    
    paired_data = df[[col1, col2]].dropna()
    
    if len(paired_data) < 3:
        st.warning("Not enough paired non-missing values to run this test (minimum 3 required).")
        return
    
    if st.button("Run Correlation Test", key="run_correlation_btn"):
        if method.startswith("Pearson"):
            coef, p_value = stats.pearsonr(paired_data[col1], paired_data[col2])
            method_name = "Pearson"
        else:
            coef, p_value = stats.spearmanr(paired_data[col1], paired_data[col2])
            method_name = "Spearman"
        
        st.markdown("#### Results")
        st.write(f"**{method_name} Correlation Coefficient:** {coef:.4f}")
        st.write(f"**P-value:** {p_value:.4f}")
        
        # Strength interpretation
        abs_coef = abs(coef)
        if abs_coef < 0.3:
            strength = "weak"
        elif abs_coef < 0.7:
            strength = "moderate"
        else:
            strength = "strong"
        
        direction = "positive" if coef > 0 else "negative"
        
        alpha = 0.05
        if p_value < alpha:
            st.error(f"**Decision:** Reject H0 (p = {p_value:.4f} < {alpha})")
            st.write(
                f"**Interpretation:** There is a statistically significant **{strength} {direction}** "
                f"correlation between '{col1}' and '{col2}' (r = {coef:.2f})."
            )
        else:
            st.success(f"**Decision:** Fail to reject H0 (p = {p_value:.4f} ≥ {alpha})")
            st.write(f"**Interpretation:** No statistically significant correlation found between '{col1}' and '{col2}'.")
        st.caption("ℹ️ Strength thresholds (weak < 0.3, moderate 0.3-0.7, strong > 0.7) are a general guideline — interpretation may vary by field.")
        st.caption("⚠️ Correlation does not imply causation — a significant relationship doesn't mean one variable causes changes in the other.")