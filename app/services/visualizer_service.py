import base64
import json
import logging
import re
from io import BytesIO

# 1. MUST BE BEFORE ANY PYPLOT IMPORTS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import seaborn as sns
import pandas as pd

from app.config import Config
from app.models import AgentState, Visualization
from app.utils.llm_utils import get_llm
from langchain.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


def visualization_check(state: AgentState) -> AgentState:
    print("=================== Visualization Checkpoint =====================")
    return state


def _clean_json_response(content: str) -> dict:
    """Safely extracts JSON from markdown-wrapped LLM text."""
    clean_text = content.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', clean_text)
    if match:
        clean_text = match.group(1).strip()
    return json.loads(clean_text)


def create_visualization(data, query, visualization_type, x_column, y_column=None, title=None, palette='viridis', style='whitegrid'):
    if not data:
        return None

    df = pd.DataFrame(data)
    if df.empty:
        return None

    plt.close('all')
    sns.set_theme(style=style)
    plt.figure(figsize=(10, 6))

    # Verify columns exist
    if x_column not in df.columns:
        x_column = df.columns[0]
    
    y_column_provided = bool(y_column and str(y_column).strip() and y_column in df.columns)

    # Date parsing check
    try:
        if pd.api.types.is_string_dtype(df[x_column]) or pd.api.types.is_object_dtype(df[x_column]):
            if any(term in x_column.lower() for term in ['date', 'time', 'year', 'month', 'day']):
                df[x_column] = pd.to_datetime(df[x_column], errors='ignore')
    except Exception:
        pass

    is_time_series = pd.api.types.is_datetime64_any_dtype(df[x_column])

    # Convert y-column to numeric if present
    if y_column_provided:
        df[y_column] = pd.to_numeric(df[y_column], errors='coerce').fillna(0)

    try:
        if visualization_type == "bar":
            if not y_column_provided and len(df.columns) > 1:
                y_column = df.columns[1]
                y_column_provided = True
                df[y_column] = pd.to_numeric(df[y_column], errors='coerce').fillna(0)
            
            ax = sns.barplot(data=df, x=x_column, y=y_column, palette=palette)
            plt.xticks(rotation=45, ha='right')

        elif visualization_type == "line":
            if not y_column_provided and len(df.columns) > 1:
                y_column = df.columns[1]
                y_column_provided = True
                df[y_column] = pd.to_numeric(df[y_column], errors='coerce').fillna(0)

            ax = sns.lineplot(data=df, x=x_column, y=y_column, marker='o', palette=palette)
            if is_time_series:
                plt.gcf().autofmt_xdate()
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        elif visualization_type == "pie":
            val_col = y_column if y_column_provided else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
            label_col = x_column
            df[val_col] = pd.to_numeric(df[val_col], errors='coerce').fillna(0)
            plt.pie(df[val_col], labels=df[label_col], autopct='%1.1f%%', startangle=90)
            plt.axis('equal')

        elif visualization_type == "scatter":
            if not y_column_provided and len(df.columns) > 1:
                y_column = df.columns[1]
            ax = sns.scatterplot(data=df, x=x_column, y=y_column)

        elif visualization_type == "histogram":
            val_col = y_column if y_column_provided else x_column
            ax = sns.histplot(data=df, x=val_col, kde=True)

        elif visualization_type == "box":
            ax = sns.boxplot(data=df, x=x_column, y=y_column)
            plt.xticks(rotation=45, ha='right')

        else:
            # Fallback to barplot
            ax = sns.barplot(data=df, x=x_column, y=y_column if y_column_provided else df.columns[-1])
            plt.xticks(rotation=45, ha='right')

        plt.title(title or f"Visualization: {query}", fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close('all')

        description = f"{visualization_type.capitalize()} chart of {x_column}" + (f" vs {y_column}" if y_column_provided else "")
        return Visualization(image=image_base64, description=description)

    except Exception as e:
        logger.error(f"Error rendering chart: {e}", exc_info=True)
        plt.close('all')
        return None


def select_visualization(state: AgentState) -> dict:
    llm = get_llm(Config.LLM_PROVIDER, Config.LLM_MODEL)

    data = state.get("execution_result").data if state.get("execution_result") else None
    if not data:
        return {"visualization_type": "bar", "x_column": "index", "y_column": None}

    df = pd.DataFrame(data)
    sample_data = df.head(5).to_dict(orient="records")
    data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

    prompt = ChatPromptTemplate.from_template("""
You are an expert data analyst.
Select the most appropriate visualization type and columns to answer the user query.

User Query: {user_query}
Data Sample (first 5 rows): {sample_data}
Data Columns & Types: {data_types}

Return ONLY a valid JSON object:
{{
    "visualization_type": "bar|line|scatter|pie|histogram|box",
    "x_column": "column_name",
    "y_column": "numeric_column_name_or_null",
    "title": "Clean Chart Title",
    "explanation": "Short justification"
}}
""")

    chain = prompt | llm

    try:
        response = chain.invoke({
            "user_query": state.get("user_query", ""),
            "sample_data": json.dumps(sample_data),
            "data_types": json.dumps(data_types)
        })

        visualization_params = _clean_json_response(response.content)
        logger.info(f"Selected visualization parameters: {visualization_params}")
        return visualization_params

    except Exception as e:
        logger.warning(f"Failed to extract structured visualization params: {e}. Using deterministic fallback.")
        cols = list(df.columns)
        return {
            "visualization_type": "bar",
            "x_column": cols[0],
            "y_column": cols[1] if len(cols) > 1 else None,
            "title": state.get("user_query", "Chart"),
            "explanation": "Default fallback visualization."
        }


# In app/services/visualizer_service.py

def data_visualizer(state: AgentState) -> AgentState:
    print("=================== Data Visualization =====================")
    exec_result = state.get("execution_result")
    eval_result = state.get("evaluation_result")

    # 1. Skip if no data or execution failed
    if not exec_result or not exec_result.success or not exec_result.data:
        state["visualization"] = None
        return state

    data = exec_result.data
    df = pd.DataFrame(data)

    # 2. Skip if the evaluator explicitly says no visualization is required
    if eval_result and not getattr(eval_result, "requires_visualization", False):
        user_query = state.get("user_query", "").lower()
        has_viz_word = any(w in user_query for w in ["plot", "chart", "graph", "visualize"])
        if not has_viz_word:
            state["visualization"] = None
            return state

    # 3. Guard: Do not plot if there are no numeric columns to measure
    # Exclude ID columns from being used as metrics
    numeric_cols = [
        c for c in df.select_dtypes(include=['number']).columns 
        if not c.lower().endswith('_id') and c.lower() != 'id'
    ]

    if not numeric_cols:
        logger.info("No numeric metric columns available for plotting. Skipping visualization.")
        state["visualization"] = None
        return state

    # 4. Generate visualization
    try:
        params = select_visualization(state)
        title = params.get("title", "")

        # Guard: Check if selector decided no visualization is needed
        if "no visualization" in title.lower() or not params.get("y_column"):
            state["visualization"] = None
            return state

        # Ensure y_column is actually numeric
        y_col = params.get("y_column")
        if y_col not in numeric_cols and not pd.api.types.is_numeric_dtype(df[y_col]):
            y_col = numeric_cols[0]

        state["visualization"] = create_visualization(
            data=data,
            query=state.get("user_query", ""),
            visualization_type=params.get("visualization_type", "bar"),
            x_column=params.get("x_column"),
            y_column=y_col,
            title=title
        )
        state["visualization_explanation"] = params.get("explanation", "")
    except Exception as e:
        logger.error(f"Failed in data_visualizer: {e}")
        state["visualization"] = None

    return state