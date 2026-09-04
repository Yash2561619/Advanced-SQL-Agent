# app/services/sql_executor_service.py

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

from app.config import Config
from app.models import AgentState, SQLExecutionResult

logger = logging.getLogger(__name__)


def _get_readonly_engine():
    db_url = Config.DATABASE_URL
    if "sqlite" in db_url:
        raw_path = db_url.replace("sqlite:///", "").split("?")[0]
        resolved_path = Path(raw_path).resolve().as_posix()
        # timeout=10 prevents hanging threads on locked files
        ro_url = f"sqlite:///file:{resolved_path}?mode=ro&uri=true"
        return create_engine(ro_url, connect_args={"timeout": 10})
    return create_engine(db_url)


def _run_query_safely(query: Optional[str]) -> SQLExecutionResult:
    """
    Executes a single SQL query against the read-only database.
    Catches multi-statement executions, schema mutations, and syntax issues.
    """
    if not query or not str(query).strip():
        return SQLExecutionResult(
            success=False,
            data=None,
            error_message="Empty or missing SQL query."
        )

    clean_query = query.strip().rstrip(';')

    try:
        engine = _get_readonly_engine()
        with engine.connect() as connection:
            # text() with read-only connection executes strictly as a single statement
            result = connection.execute(text(clean_query))
            
            if result.returns_rows:
                rows = result.fetchall()
                df = pd.DataFrame(rows, columns=result.keys())
                records = df.to_dict(orient="records")
            else:
                records = []

            return SQLExecutionResult(
                success=True,
                data=records,
                error_message=None
            )

    except sqlite3.OperationalError as e:
        error_msg = f"Database driver security rejection: {str(e)}"
        logger.error(error_msg)
        return SQLExecutionResult(success=False, data=None, error_message=error_msg)

    except Exception as e:
        error_msg = f"Query execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return SQLExecutionResult(success=False, data=None, error_message=error_msg)


def execute_sql(state: AgentState) -> AgentState:
    print("============= Executing SQL Code ==============")
    gen_sql = state.get("generated_sql")
    query = getattr(gen_sql, "sql_query", None) if gen_sql else None
    state["execution_result"] = _run_query_safely(query)
    return state


def execute_sql_reflection(state: AgentState) -> AgentState:
    print("============= [Reflection] Executing SQL Code ==============")
    refl_sql = state.get("reflected_generated_sql")
    query = getattr(refl_sql, "reflected_sql_query", None) if refl_sql else None
    state["execution_result"] = _run_query_safely(query)
    return state


def execute_sql_corrected(state: AgentState) -> AgentState:
    print("============= [Correction] Executing SQL Code ==============")
    corr_sql = state.get("sql_correction")
    query = getattr(corr_sql, "corrected_sql_query", None) if corr_sql else None
    state["execution_result"] = _run_query_safely(query)
    return state