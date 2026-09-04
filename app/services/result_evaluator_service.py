# app/services/result_evaluator_service.py

import logging
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from app.config import Config
from app.models import AgentState, EvaluationResult
from app.services.session_service import SessionService
from app.utils.json_utils import process_node_output
from app.utils.llm_utils import get_llm

logger = logging.getLogger(__name__)

VIZ_KEYWORDS = {
    "plot", "chart", "graph", "visualize", "bar", "line",
    "scatter", "trend", "histogram", "distribution", "pie"
}


def result_evaluator(state: AgentState) -> AgentState:
    print("================= Evaluating the results =================")
    logger.info("Entering result evaluator")

    exec_result = state.get("execution_result")

    # 1. Execution failure handler
    if not exec_result or not getattr(exec_result, "success", False):
        err_msg = getattr(exec_result, "error_message", "Unknown execution error") if exec_result else "No execution result"
        state["evaluation_result"] = EvaluationResult(
            is_result_relevant=False,
            explanation=f"Query execution failed: {err_msg}",
            requires_visualization=False,
            summary="The query execution failed, so no results are available to summarize."
        )
        state["is_result_relevant"] = False
        state["reflection"] = {
            "type": "result_evaluation",
            "issue": "Query execution failed",
            "suggestion": "Review and fix the SQL query execution error"
        }
        return state

    rows = getattr(exec_result, "data", []) or []
    user_query = state.get("user_query", "")
    if not user_query:
        analyzed = state.get("analyzed_query")
        user_query = getattr(analyzed, "original_query", "") if analyzed else ""

    query_lower = user_query.lower()
    has_explicit_viz = any(word in query_lower for word in VIZ_KEYWORDS)

    # 2. Fast-Path Optimization:
    # If the query executed cleanly and returned rows, bypass the LLM roundtrip.
    # This prevents 429 rate limit errors on Groq and reduces latency.
    if len(rows) > 0:
        # Require visualization if explicitly asked or if multiple rows can be charted
        requires_viz = has_explicit_viz or (len(rows) >= 2 and any(
            isinstance(v, (int, float)) for r in rows for v in r.values() if v is not None
        ))

        state["evaluation_result"] = EvaluationResult(
            is_result_relevant=True,
            explanation="Query executed successfully with data rows returned.",
            requires_visualization=requires_viz,
            summary="Data retrieved successfully."
        )
        state["is_result_relevant"] = True
        state["reflection"] = None

        logger.info(f"Fast-path evaluation: Result relevance: True, Requires visualization: {requires_viz}")
        return state

    # 3. Empty Result Handler: Run LLM evaluation or reflect when rows == 0
    df = pd.DataFrame(rows)
    results_summary = df.describe().to_string() if not df.empty else "No results (0 rows returned)"

    session_id = state.get("session_id")
    session_history_summary = "No session history available"
    if session_id:
        try:
            session_history = SessionService.get_session_history(session_id=session_id)
            if session_history:
                session_history_summary = "\n".join(
                    [f"Query: {item.query}\nResponse: {item.response}" for item in session_history]
                )
        except Exception as e:
            logger.error(f"Error fetching session history: {str(e)}", exc_info=True)
            session_history_summary = "Error fetching session history"

    llm = get_llm(Config.LLM_PROVIDER, Config.LLM_MODEL)

    prompt = ChatPromptTemplate.from_template("""
        Given the following:
        1. Original user query: {original_query}
        2. Analyzed query: {analyzed_query}
        3. Generated SQL query: {generated_sql}
        4. Query results summary:
        {results_summary}
        5. Session history:
        {session_history}

        Task 1: Evaluate the relevance and quality of the query results to the original user query.
        Task 2: If the query returned 0 rows or is not relevant, provide explanation and suggestions on how to improve the SQL query.
        Task 3: Determine whether the results require visualization. Queries with 0 rows should NOT require visualization.
        Task 4: Summarize the findings in a concise, user-friendly manner.

        Respond in the following JSON format:
        {{
            "is_result_relevant": true/false,
            "explanation": "Detailed explanation of your evaluation",
            "improvement_suggestion": "Suggestion on how to improve the SQL query if not relevant",
            "requires_visualization": true/false,
            "summary": "Your human-friendly summary here"
        }}
    """)

    analyzed_query_obj = state.get("analyzed_query")
    generated_sql_obj = state.get("generated_sql")

    chain = prompt | llm

    try:
        response = chain.invoke({
            "original_query": user_query,
            "analyzed_query": getattr(analyzed_query_obj, "analyzed_query", ""),
            "generated_sql": getattr(generated_sql_obj, "sql_query", ""),
            "results_summary": results_summary,
            "session_history": session_history_summary
        })

        parsed_response = process_node_output(response.content, "result_evaluator")

        is_relevant = bool(parsed_response.get('is_result_relevant', False))
        requires_viz = bool(parsed_response.get('requires_visualization', False)) and len(rows) > 0

        state["evaluation_result"] = EvaluationResult(
            is_result_relevant=is_relevant,
            explanation=parsed_response.get('explanation', "Evaluation complete."),
            requires_visualization=requires_viz,
            summary=parsed_response.get('summary', "Summary generated.")
        )
        state["is_result_relevant"] = is_relevant

        if not is_relevant:
            state["reflection"] = {
                "type": "result_evaluation",
                "issue": "Results not relevant or returned no rows",
                "suggestion": parsed_response.get('improvement_suggestion', "Review SQL filter predicates and join keys")
            }
        else:
            state["reflection"] = None

    except Exception as e:
        logger.error(f"Error in result evaluator LLM: {str(e)}", exc_info=True)
        state["evaluation_result"] = EvaluationResult(
            is_result_relevant=False,
            explanation=f"An error occurred during evaluation: {str(e)}",
            requires_visualization=False,
            summary="Failed to evaluate results due to an error."
        )
        state["is_result_relevant"] = False
        state["reflection"] = {
            "type": "result_evaluation",
            "issue": "Error during evaluation",
            "suggestion": "Review and fix the evaluation process"
        }

    logger.info(f"Result relevance: {state.get('is_result_relevant')}")
    logger.info(f"Requires visualization: {state['evaluation_result'].requires_visualization}")

    return state