import sqlite3
import pandas as pd


def create_sql_table(df : pd.DataFrame, table_name: str = "dataset") -> sqlite3.Connection :
    """Converts a DataFrame into an in-memory SQLite table.
    Returns the connection object so queries can be run against it."""

    conn = sqlite3.connect(":memory:")
    df.to_sql(table_name,conn,index=False,if_exists='replace')
    return conn


def run_sql_query(conn,query:str)-> pd.DataFrame :
    """Executes a SQL query against the given SQLite connection.
    Returns results as a DataFrame, or an error message DataFrame if it fails."""

    try:
        result_df = pd.read_sql_query(query,conn)
        return result_df
    except Exception as e:
        return pd.DataFrame({"Error": [f"Could not run query: {e}"]})