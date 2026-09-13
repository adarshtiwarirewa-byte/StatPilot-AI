import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import pandas as pd


load_dotenv()
api_key=os.getenv("GROQ_API_KEY")  or st.secrets.get("GROQ_API_KEY")

client = Groq(api_key=api_key)


def build_schema_description(df:pd.DataFrame) -> str:
    """
    Builds a column-by-column description including data type and
    sample/unique values, so the LLM knows the actual value format.
    """

    lines = []
    for col in df.columns:
        dtype = df[col].dtype
        nunique = df[col].nunique()

        if nunique <= 15 :
            sample_values = df[col].dropna().unique().tolist()
            lines.append(f'-"{col}" ({dtype}) : possible values = {sample_values}')
        else:
            sample_values = df[col].dropna().unique().tolist()[:5]
            lines.append(f'-"{col}" ({dtype}) : possible values = {sample_values}')


    return "\n".join(lines)




def generate_sql_query(question: str, table_name: str, df:pd.DataFrame) -> str:
    """
    Uses Groq LLM to convert a natural language question into a SQL query.
    """
    
    schema_description = build_schema_description(df)

    prompt = f"""You are a SQL expert. Convert the following natural language question into a valid SQLite SQL query.

    Table name: {table_name}
    Column details (name, type, and actual sample values in the data):
    {schema_description}

    Question: {question}

    Rules:
    - Return ONLY the SQL query, nothing else.
    - No explanations, no markdown, no code fences.
    - Use only the table and columns listed above.
    - Always wrap every column name in double quotes, exactly as given (e.g. "1st_Road_Class", "Local_Authority_(District)"), since column names may contain spaces, parentheses, or start with numbers
    - Match the exact value format shown in the sample values above (e.g. if a column shows values like [0, 1], use 0/1 not 'Yes'/'No'; if it shows ['Yes', 'No'], use those exact strings).
    - If the question cannot be answered with the given columns, return: SELECT 'Cannot answer this question with available data' AS message;

    SQL query:"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    sql_query = response.choices[0].message.content.strip()

    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

    return sql_query


def generate_update_sql(instruction: str, table_name: str, df: pd.DataFrame) -> str:
    """
    Uses Groq LLM to convert a natural language data-cleaning instruction
    into a single, safe SQLite UPDATE statement.
    """

    schema_description = build_schema_description(df)

    prompt = f"""You are a SQL expert helping with data cleaning. Convert the following instruction into a single valid SQLite UPDATE statement.

    Table name: {table_name}
    Column details (name, type, and actual sample values in the data):
    {schema_description}

    Instruction: {instruction}

    Rules:
    - Return ONLY one SQL statement, nothing else.
    - No explanations, no markdown, no code fences.
    - Only generate UPDATE statements. Never generate DROP, DELETE, TRUNCATE, ALTER, ATTACH, DETACH, or PRAGMA statements.
    - Always wrap every column name in double quotes, exactly as given (e.g. "1st_Road_Class", "Local_Authority_(District)").
    - When converting text to numeric, never rely on CAST() alone, since SQLite's CAST silently returns 0 for non-numeric text.
      Instead, first set non-numeric values to NULL explicitly using a CASE WHEN with GLOB pattern matching
      (e.g. CASE WHEN "col" GLOB '-?[0-9]*' OR "col" GLOB '[0-9]*' THEN CAST("col" AS REAL) ELSE NULL END),
      so invalid values become NULL, not 0.
    - If the instruction cannot be safely expressed as a single UPDATE statement, return: SELECT 'Cannot generate a safe update for this instruction' AS message;

    SQL query:"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    sql_query = response.choices[0].message.content.strip()
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

    return sql_query



BLOCKED_KEYWORDS = ["DROP", "DELETE", "TRUNCATE", "ALTER", "ATTACH", "DETACH", "PRAGMA", "--", ";--"]


def is_query_safe(sql_query: str) -> tuple[bool, str | None]:
    """
    Checks a generated SQL query against a blocklist of destructive/unsafe keywords.
    Returns (is_safe, blocked_keyword_if_any).
    """
    query_upper = sql_query.upper()

    if not query_upper.strip().startswith("UPDATE") and not query_upper.strip().startswith("SELECT"):
        return False, "non-UPDATE/SELECT statement"

    for keyword in BLOCKED_KEYWORDS:
        if keyword in query_upper:
            return False, keyword

    return True, None