import json
import logging
import time
from flask import Blueprint, request, jsonify, render_template, Response, stream_with_context

from app.config import Config
from app.models import AgentState
from app.services.graph_service import create_analysis_graph
from app.services.session_service import SessionService
from app import memory_service

main_bp = Blueprint('main', __name__)
logger = logging.getLogger(__name__)


def _build_initial_state(user_query: str, session_id: str, run_id: str) -> AgentState:
    """Helper to initialize a clean AgentState payload."""
    recent_history = SessionService.get_recent_history(session_id, limit=5)
    relevant_memories = memory_service.search_memory(user_query)

    return AgentState(
        user_query=user_query,
        db_info=None,
        analyzed_query=None,
        generated_sql=None,
        validation_result=None,
        execution_result=None,
        evaluation_result=None,
        visualization=None,
        summary=None,
        error=None,
        is_query_relevant=False,
        is_result_relevant=False,
        regenerate_list=[],
        reanalyze_list=[],
        reflection=None,
        reflected_generated_sql=None,
        relevant_memories=relevant_memories,
        session_id=session_id,
        run_id=run_id,
        recent_history=recent_history,
        sql_correction=None,
        visualization_explanation=None
    )


def _format_visualization(viz_obj):
    """Safely extracts image and description whether viz is a dict or Pydantic model."""
    if not viz_obj:
        return None
    if isinstance(viz_obj, dict):
        return {
            "image": viz_obj.get("image"),
            "description": viz_obj.get("description")
        }
    return {
        "image": getattr(viz_obj, "image", None),
        "description": getattr(viz_obj, "description", None)
    }


def _record_interaction(session_id: str, run_id: str, query: str, summary: str):
    """Stores the round-trip conversation into vector memory and relational history."""
    try:
        memory_service.add_memory(
            text=f"Query: {query}\nResponse: {summary}",
            metadata={"session_id": session_id, "run_id": run_id}
        )
    except Exception as e:
        logger.warning(f"Failed to record memory: {e}")

    try:
        SessionService.add_to_session_history(
            session_id=session_id,
            run_id=run_id,
            query=query,
            response=summary
        )
    except Exception as e:
        logger.warning(f"Failed to record session history: {e}")


@main_bp.route('/')
def index():
    return render_template('index.html')


@main_bp.route('/analyze', methods=['POST'])
def analyze_query():
    start_time = time.time()
    data = request.get_json() or {}
    user_query = data.get('query')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        session_id = SessionService.get_or_create_session()
        run_id = SessionService.create_run()
        analysis_graph = create_analysis_graph(memory_service)
        initial_state = _build_initial_state(user_query, session_id, run_id)

        final_state = analysis_graph.invoke(initial_state)

        summary = final_state.get('summary', "No summary available.")
        response = {
            "summary": summary,
            "visualization": _format_visualization(final_state.get('visualization'))
        }

        _record_interaction(session_id, run_id, user_query, summary)
        logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in analyze_query: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@main_bp.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()
    data = request.get_json() or {}
    user_query = data.get('query')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        session_id = SessionService.get_or_create_session()
        run_id = SessionService.create_run()
        analysis_graph = create_analysis_graph(memory_service)
        initial_state = _build_initial_state(user_query, session_id, run_id)

        final_state = analysis_graph.invoke(initial_state)

        if not final_state:
            return jsonify({"error": "No result generated"}), 500

        summary = final_state.get('summary', "No summary available.")
        response = {
            "summary": summary,
            "visualization": _format_visualization(final_state.get('visualization'))
        }

        _record_interaction(session_id, run_id, user_query, summary)
        logger.info(f"Chat query processed in {time.time() - start_time:.2f}s")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@main_bp.route('/stream', methods=['POST'])
def stream_chat():
    data = request.get_json() or {}
    user_query = data.get('query')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    def generate():
        try:
            session_id = SessionService.get_or_create_session()
            run_id = SessionService.create_run()
            analysis_graph = create_analysis_graph(memory_service)
            initial_state = _build_initial_state(user_query, session_id, run_id)

            last_state = initial_state
            for output in analysis_graph.stream(initial_state):
                # LangGraph stream yields {node_name: updated_state}
                for node_name, state_update in output.items():
                    if isinstance(state_update, dict):
                        last_state.update(state_update)
                    yield json.dumps({"type": "update", "node": node_name}) + "\n"

            summary = last_state.get('summary', "No summary available.")
            final_response = {
                "summary": summary,
                "visualization": _format_visualization(last_state.get('visualization'))
            }

            _record_interaction(session_id, run_id, user_query, summary)
            yield json.dumps({"type": "final", "content": final_response}) + "\n"

        except Exception as e:
            logger.error(f"Error in stream_chat: {str(e)}", exc_info=True)
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

    return Response(stream_with_context(generate()), content_type='application/x-ndjson')