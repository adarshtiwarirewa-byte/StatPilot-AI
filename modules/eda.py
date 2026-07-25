import pandas as pd
import plotly.express as px



def get_numeric_columns(df: pd.DataFrame) -> list:
    """
    Returns list of numeric column names from the dataframe.
    """
    return df.select_dtypes(include="number").columns.tolist()

def plot_histogram(df:pd.DataFrame, column:str):
    "Returns a Plotly histogram figure for the given numerical column."

    fig = px.histogram(df,x=column,title=f"Distribution of {column}",nbins=30)
    fig.update_layout(bargap=0.1)
    return fig


def plot_correlation_heatmap(df:pd.DataFrame, numeric_columns:list):
    """Returns a Plotly heatmap figure showing correlation between numeric columns."""

    if len(numeric_columns)<2 :
        return None
    
    corr_matrix = df[numeric_columns].corr()

    fig = px.imshow(corr_matrix,text_auto=".2f",aspect='auto',color_continuous_scale='RdBu_r',zmin=-1,zmax=1)

    return fig




def plot_boxplot(df: pd.DataFrame, column: str):
    """
    Returns a Plotly box plot figure for the given numeric column.
    """
    fig = px.box(df, y=column, title=f"Box Plot of {column}")
    return fig


def plot_violin(df: pd.DataFrame, column: str):
    """
    Returns a Plotly violin plot figure for the given numeric column.
    """
    fig = px.violin(df, y=column, box=True, points="outliers", title=f"Violin Plot of {column}")
    return fig


def plot_scatter(df: pd.DataFrame, x_column: str, y_column: str):
    """
    Returns a Plotly scatter plot figure between two numeric columns.
    """
    fig = px.scatter(df, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
    return fig



def get_categorical_columns(df:pd.DataFrame) -> list :
    """
    Returns list of categorical (non numeric) columns from datafrane.
    """
    return df.select_dtypes(include=['object','category']).columns.to_list()


def plot_bar_chart(df: pd.DataFrame, column: str, top_n: int = 15):
    """
    Returns a Plotly bar chart showing value counts for a categorical column.
    """
    value_counts = df[column].value_counts().head(top_n).reset_index()
    value_counts.columns = [column, "count"]

    fig = px.bar(
        value_counts,
        x=column,
        y="count",
        title=f"Frequency of {column}" + (f" (Top {top_n})" if df[column].nunique() > top_n else ""),
    )
    return fig


def plot_pie_chart(df:pd.DataFrame,column:str,top_n:int = 10):
    """
    Returns a Plotly pie chart showing proportion of categories for a categorical column.
    """

    value_counts = df[column].value_counts().head(top_n).reset_index()
    value_counts.columns = [column,"count"]

    fig = px.pie(
        value_counts,names=column,values="count",
        title=f"Proportion of {column}" + (f" (Top {top_n})" if df[column].nunique() > top_n else ""),
        )

    return fig


def get_frequency_table(df:pd.DataFrame,column:str):
    """
    Returns a frequency table (count + percentage) for a categorical column.
     """
    
    counts = df[column].value_counts()
    percentages = (counts/counts.sum() *100).round(2)

    table = pd.DataFrame({column:counts.index,"Count":counts.values,"Percentage":percentages.values})
    return table