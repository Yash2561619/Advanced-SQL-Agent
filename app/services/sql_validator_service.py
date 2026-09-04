# app/services/sql_validator_service.py

import json
import logging
import sqlparse
from langchain.prompts import ChatPromptTemplate
from app.config import Config
from app.models import AgentState, SQLValidationResult
from app.utils.json_utils import process_node_output
from app.utils.llm_utils import get_llm

logger = logging.getLogger(__name__)

FORBIDDEN_KEYWORDS = {
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
    "REPLACE", "CREATE", "GRANT", "REVOKE", "ATTACH", "DETACH",
    "PRAGMA", "VACUUM", "REINDEX"
}


def validate_sql_safety(sql_query: str) -> tuple[bool, str]:
    """
    Deterministic security check:
    1. Rejects multi-statement stacked queries (semicolon piggybacking).
    2. Enforces top-level SELECT statements only.
    3. Traverses the AST for any DDL/DML mutation keywords.
    """
    if not sql_query or not str(sql_query).strip():
        return False, "Query string is empty."

    # Parse query into discrete statements
    parsed = sqlparse.parse(str(sql_query).strip())

    # 1. Reject multiple statements (uses str(s) instead of deprecated s.to_unicode())
    statements = [s for s in parsed if str(s).strip().strip(';')]
    if len(statements) != 1:
        return False, f"Rejected: Multi-statement query detected ({len(statements)} statements found). Only single queries are allowed."

    stmt = statements[0]

    # 2. Strict statement type check
    stmt_type = stmt.get_type()
    if stmt_type != "SELECT":
        return False, f"Rejected: Statement type '{stmt_type}' is forbidden. Only 'SELECT' queries are permitted."

    # 3. Deep token AST inspection for prohibited tokens
    for token in stmt.flatten():
        tok_val = token.normalized.upper() if hasattr(token, "normalized") else str(token).upper().strip()
        if tok_val in FORBIDDEN_KEYWORDS:
            return False, f"Rejected: Forbidden keyword '{tok_val}' detected."

    return True, "Safe read-only query."


def sql_validator(state: AgentState) -> AgentState:
    print("============== Validating SQL Code ================")

    generated_sql = state.get("generated_sql")
    query_str = getattr(generated_sql, "sql_query", "") if generated_sql else ""

    # Phase 1: Hard Deterministic Security Filter
    is_safe, security_reason = validate_sql_safety(query_str)
    if not is_safe:
        logger.warning(f"Deterministic SQL validation failed: {security_reason}")
        state["validation_result"] = SQLValidationResult(
            is_sql_valid=False,
            issues=[security_reason],
            suggested_fix="Rewrite query as a single, read-only SELECT statement without prohibited keywords."
        )
        state["reflection"] = {
            "type": "sql_validation",
            "issues": [security_reason],
            "suggested_fix": "Rewrite query as a single, read-only SELECT statement without prohibited keywords."
        }
        return state

    # Phase 2: LLM Logic, Schema, and Syntax Validation
    llm = get_llm(Config.LLM_PROVIDER, Config.LLM_MODEL)

    prompt = ChatPromptTemplate.from_template("""
        Given the following:
        1. Original user query: {original_query}
        2. Analyzed query: {analyzed_query}
        3. Generated SQL query: {sql_query}
        4. SQL query explanation: {sql_explanation}
        5. Selected tables and their schemas:
        {table_schemas}

        Task 1: Validate the SQL query for correctness, logical accuracy, and relevance to the original query.
        Task 2: Provide suggestions to improve the Generated SQL query if issues are found.

        Please check for the following:
        1. SQL syntax errors
        2. Use of non-existent tables or columns against the provided schemas
        3. Logical errors in JOIN conditions or WHERE clauses
        4. Appropriate use of aggregation functions and GROUP BY clauses
        5. Relevance of the query to the original user question
        6. Potential performance bottlenecks (e.g., missing LIMIT on large tables)

        Respond in the following JSON format:
        {{
            "is_sql_valid": true/false,
            "issues": ["issue1", "issue2"],
            "suggested_fix": "Suggested SQL query fix or improvements"
        }}

        If the SQL query is valid, set is_sql_valid to true and leave issues and suggested_fix empty.
        If issues are found, set is_sql_valid to false, list the issues, and provide a suggested fix.
    """)

    # Safely extract selected table schemas
    selected_table_schemas = {}
    db_info = state.get("db_info", {})
    analyzed_query = state.get("analyzed_query")
    selected_tables = getattr(analyzed_query, "selected_tables", []) if analyzed_query else []

    for name in selected_tables:
        if name in db_info and hasattr(db_info[name], "table_schema"):
            selected_table_schemas[name] = db_info[name].table_schema.dict()

    chain = prompt | llm

    try:
        response = chain.invoke({
            "original_query": getattr(analyzed_query, "original_query", ""),
            "analyzed_query": getattr(analyzed_query, "analyzed_query", ""),
            "sql_query": query_str,
            "sql_explanation": getattr(generated_sql, "explanation", ""),
            "table_schemas": json.dumps(selected_table_schemas, indent=2)
        })

        validation_result = process_node_output(response.content, "sql_validator")

        is_sql_valid = validation_result.get('is_sql_valid')
        if is_sql_valid is None:
            logger.warning("Validation result did not contain 'is_sql_valid' field. Assuming SQL is invalid.")
            is_sql_valid = False

        state["validation_result"] = SQLValidationResult(
            is_sql_valid=bool(is_sql_valid),
            issues=validation_result.get('issues', []),
            suggested_fix=validation_result.get('suggested_fix', '')
        )

        if is_sql_valid:
            state["reflection"] = None
        else:
            state["reflection"] = {
                "type": "sql_validation",
                "issues": validation_result.get('issues', []),
                "suggested_fix": validation_result.get('suggested_fix', '')
            }

    except Exception as e:
        logger.error(f"Error in LLM SQL validation: {str(e)}", exc_info=True)
        state["validation_result"] = SQLValidationResult(
            is_sql_valid=False,
            issues=["Unexpected error in validation process"],
            suggested_fix="Please review the SQL query and try again."
        )
        state["reflection"] = {
            "type": "sql_validation",
            "issues": ["Unexpected error in validation process"],
            "suggested_fix": "Please review the SQL query and try again."
        }

    return state