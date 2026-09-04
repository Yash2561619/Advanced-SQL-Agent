import json
import logging
from langchain.prompts import ChatPromptTemplate
from app.models import AgentState
from app.utils.llm_utils import get_llm
from app.config import Config

logger = logging.getLogger(__name__)


def summarizer_node(state: AgentState) -> AgentState:
    print("=================== Summarization =====================")

    analyzed_query = state.get("analyzed_query")
    if not analyzed_query or not getattr(analyzed_query, "is_query_relevant", False):
        explanation = getattr(analyzed_query, "explanation", "Query evaluation failed.") if analyzed_query else "Query evaluation failed."
        state['summary'] = (
            f"I'm sorry, but your query '{state.get('user_query', '')}' "
            f"is not relevant to the available database information. {explanation}"
        )
        return state

    llm = get_llm(Config.LLM_PROVIDER, Config.LLM_MODEL)

    prompt = ChatPromptTemplate.from_template("""You are an expert business intelligence and database analyst.

Given the analytical findings:
1. User Query: {user_query}
2. Interpreted Task: {analyzed_query}
3. SQL Executed:
{sql_query}
4. Query Results (JSON data):
{execution_result}
5. Evaluation / Context:
{evaluation_result}

Instructions:
- Provide a clear, insightful answer to the user's question based on the retrieved data.
- Format all tabular data cleanly using standard Markdown tables with proper header separators (`|---|---|`) and clean row breaks.
- Ensure every table row is on its own separate line. Do not combine table rows into a single paragraph.
- Include distinct double line breaks between paragraphs and bullet points so Markdown parsers render them cleanly.
- If the result set is empty or contains no rows (`[]`), state clearly that no records matched and suggest why based on the table schema.
- Highlight key trends, top performers, or aggregate figures concisely.
""")

    chain = prompt | llm | (lambda x: x.content)

    exec_result = state.get('execution_result')
    eval_result = state.get('evaluation_result')
    gen_sql = state.get('generated_sql')

    # Safely extract SQL text
    sql_text = "N/A"
    if gen_sql:
        sql_text = getattr(gen_sql, "sql_query", str(gen_sql))

    # Safely extract execution data and prevent JSON serialization crashes
    execution_data_str = "[]"
    if exec_result and hasattr(exec_result, "data") and exec_result.data is not None:
        execution_data_str = json.dumps(exec_result.data, default=str)

    # Safely extract evaluation explanation
    eval_explanation = "N/A"
    if eval_result:
        eval_explanation = getattr(eval_result, "explanation", str(eval_result))

    # Safely extract analyzed query text
    analyzed_query_text = getattr(analyzed_query, "analyzed_query", str(analyzed_query))

    try:
        response = chain.invoke({
            "user_query": state.get('user_query', ''),
            "analyzed_query": analyzed_query_text,
            "sql_query": sql_text,
            "execution_result": execution_data_str,
            "evaluation_result": eval_explanation
        })
        state['summary'] = response
    except Exception as e:
        logger.error(f"Error generating summary: {e}", exc_info=True)
        state['summary'] = f"Data was retrieved successfully, but summarizing the results failed: {str(e)}"

    return state