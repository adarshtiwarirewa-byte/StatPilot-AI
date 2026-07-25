import os
from groq import Groq
from dotenv import load_dotenv
from numpy import dtype
from pandas import DataFrame
import pandas as pd
from pyparsing import line

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


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
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    sql_query = response.choices[0].message.content.strip()

    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

    return sql_query
