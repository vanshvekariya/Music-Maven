"""LangGraph-based Multi-Agent Orchestrator for Music Maven"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from enum import Enum
import operator

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from loguru import logger

from .sql_agent import SQLAgent
from .vector_agent import VectorAgent, infer_lang_filter_from_query
from .query_router import QueryRouter, KGQueryRouter, QueryType
from ..config.settings import get_settings


class AgentState(TypedDict):
    """
    State object for the multi-agent workflow.
    Tracks the query, routing decisions, and results from each agent.
    """
    query: str
    routing_info: Dict[str, Any]
    kg_result: Optional[Dict[str, Any]]
    sql_result: Optional[Dict[str, Any]]
    vector_result: Optional[Dict[str, Any]]
    final_response: Optional[str]
    error: Optional[str]
    metadata: Dict[str, Any]
    max_results: int
    lang_filter: Optional[str]
    use_kg: bool
    conversation_context: Optional[str]


class WorkflowStage(str, Enum):
    """Enum for workflow stages"""
    ROUTE = "route"
    KG_AGENT = "kg_agent"
    SQL_AGENT = "sql_agent"
    VECTOR_AGENT = "vector_agent"
    SYNTHESIZE = "synthesize"
    END = "end"


class MultiAgentOrchestrator:
    """
    LangGraph-based orchestrator that manages multiple agents.

    Workflow:
    1. Route query via local KG-aware classifier (zero LLM calls)
    2. If KG_DIRECT: answer from Knowledge Graph instantly
    3. If SQL: delegate to SQL agent (LLM-powered)
    4. If VECTOR: delegate to Vector agent (local embeddings)
    5. If HYBRID: KG handles structured part, Vector handles semantic part
    6. Synthesize and return
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        enable_sql: bool = True,
        enable_vector: bool = True,
        kg_engine=None,
    ):
        settings = get_settings()

        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.llm_model
        self.base_url = settings.openai_base_url
        self.kg_engine = kg_engine

        if not self.api_key:
            raise ValueError(
                "API key is required. Set OPENAI_API_KEY in environment or .env file"
            )

        self.enable_sql = enable_sql
        self.enable_vector = enable_vector

        self._initialize_router()
        self._initialize_agents()
        self._initialize_llm()
        self._build_graph()

        logger.info("Multi-Agent Orchestrator initialized")

    def _initialize_router(self) -> None:
        """Initialize the local KG-aware router (zero LLM calls)."""
        try:
            artist_names = set()
            genre_names = set()
            if self.kg_engine is not None:
                G = self.kg_engine.G
                artist_names = set(G.graph.get("artist_lookup", {}).keys())
                genre_names = set(G.graph.get("genre_lookup", {}).keys())

            self.router = KGQueryRouter(
                kg_engine=self.kg_engine,
                artist_names=artist_names,
                genre_names=genre_names,
            )
            logger.info("KG Query Router initialized (local, zero-LLM)")
        except Exception as e:
            logger.warning(f"KG router init failed, falling back to LLM router: {e}")
            self.router = QueryRouter(api_key=self.api_key, model=self.model)

    def _initialize_agents(self) -> None:
        """Initialize all agents"""
        self.agents = {}

        if self.enable_sql:
            try:
                self.agents['sql'] = SQLAgent(
                    api_key=self.api_key,
                    model=self.model
                )
                logger.info("SQL Agent initialized")
            except Exception as e:
                logger.warning(f"SQL Agent initialization failed: {e}")
                self.enable_sql = False

        if self.enable_vector:
            try:
                self.agents['vector'] = VectorAgent(
                    api_key=self.api_key,
                    model=self.model
                )
                logger.info("Vector Agent initialized")
            except Exception as e:
                logger.warning(f"Vector Agent initialization failed: {e}")
                self.enable_vector = False

        if not self.agents and self.kg_engine is None:
            raise RuntimeError("No agents could be initialized and KG is unavailable")

    def _initialize_llm(self) -> None:
        """Initialize LLM for response synthesis"""
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=0.3,
            api_key=self.api_key,
            base_url=self.base_url
        )
        logger.info("Synthesis LLM initialized")

    def _build_graph(self) -> None:
        """Build LangGraph workflow with KG support."""
        workflow = StateGraph(AgentState)

        workflow.add_node("route", self._route_query)
        workflow.add_node("kg_agent", self._execute_kg_agent)
        workflow.add_node("sql_agent", self._execute_sql_agent)
        workflow.add_node("vector_agent", self._execute_vector_agent)
        workflow.add_node("synthesize", self._synthesize_response)

        workflow.set_entry_point("route")

        workflow.add_conditional_edges(
            "route",
            self._routing_decision,
            {
                "kg_direct": "kg_agent",
                "sql": "sql_agent",
                "vector": "vector_agent",
                "hybrid": "kg_agent",
                "both_sql_vector": "sql_agent",
                "end": "synthesize",
            }
        )

        # KG agent → check if hybrid needs vector follow-up
        workflow.add_conditional_edges(
            "kg_agent",
            self._check_kg_hybrid,
            {
                "vector_agent": "vector_agent",
                "synthesize": "synthesize",
            }
        )

        # SQL agent → check if hybrid needs vector follow-up
        workflow.add_conditional_edges(
            "sql_agent",
            self._check_if_hybrid,
            {
                "vector_agent": "vector_agent",
                "synthesize": "synthesize",
            }
        )

        workflow.add_edge("vector_agent", "synthesize")
        workflow.add_edge("synthesize", END)

        self.graph = workflow.compile()
        logger.info("LangGraph workflow built successfully (with KG support)")

    # ── routing ──────────────────────────────────────────────────────────

    def _route_query(self, state: AgentState) -> AgentState:
        logger.info(f"Routing query: {state['query']}")
        try:
            routing_info = self.router.route_query(state['query'])
            # UI/API override to disable KG for SQL/vector testing.
            if not state.get("use_kg", True):
                original_agents = routing_info.get("agents", [])
                filtered_agents = [a for a in original_agents if a != "kg_direct"]

                # If KG was the only route, fall back to non-KG agents.
                if not filtered_agents:
                    if self.enable_sql and "sql" in self.agents:
                        filtered_agents = ["sql"]
                        routing_info["classification"]["type"] = "sql"
                    elif self.enable_vector and "vector" in self.agents:
                        filtered_agents = ["vector"]
                        routing_info["classification"]["type"] = "vector"

                routing_info["agents"] = filtered_agents
            state['routing_info'] = routing_info
            logger.info(
                f"Query routed to: {routing_info['agents']} "
                f"(type: {routing_info['classification']['type']})"
            )
        except Exception as e:
            logger.error(f"Routing error: {e}")
            state['error'] = f"Routing failed: {str(e)}"
            state['routing_info'] = {'agents': [], 'classification': {'type': 'unknown'}}
        return state

    def _routing_decision(self, state: AgentState) -> str:
        if state.get('error'):
            return "end"

        agents = state['routing_info'].get('agents', [])
        qtype = state['routing_info'].get('classification', {}).get('type', 'unknown')
        strategy = state['routing_info'].get('execution_strategy', '')

        if not agents:
            return "end"

        if 'kg_direct' in agents and qtype == 'kg_direct':
            return "kg_direct"

        if qtype == 'hybrid' and 'kg_direct' in agents and 'vector' in agents:
            return "hybrid"

        if 'sql' in agents and 'vector' in agents:
            if not self.enable_vector:
                return "sql"
            return "both_sql_vector"

        if 'sql' in agents:
            return "sql"

        if 'vector' in agents:
            if not self.enable_vector:
                return "sql"
            return "vector"

        if 'kg_direct' in agents:
            return "kg_direct"

        return "end"

    # ── KG agent ─────────────────────────────────────────────────────────

    def _execute_kg_agent(self, state: AgentState) -> AgentState:
        logger.info("Executing KG Agent")
        try:
            if self.kg_engine is None:
                state['kg_result'] = {'success': False, 'error': 'KG not available'}
                return state

            result = self.kg_engine.try_answer(state['query'])
            if result:
                state['kg_result'] = {
                    'success': True,
                    'data': {
                        'answer': result['answer'],
                        'query_type': 'kg_direct',
                        'source': 'knowledge_graph',
                        'template': result['template'],
                    },
                    'metadata': {'template': result['template']},
                }
                logger.info(f"KG Agent answered via template: {result['template']}")
            else:
                logger.warning("KG Agent could not answer, will fall through to synthesize")
                state['kg_result'] = {
                    'success': False,
                    'error': 'No KG template matched',
                }
        except Exception as e:
            logger.error(f"KG Agent error: {e}")
            state['kg_result'] = {'success': False, 'error': str(e)}
        return state

    def _check_kg_hybrid(self, state: AgentState) -> str:
        """After KG agent, check if we also need vector for hybrid."""
        qtype = state['routing_info'].get('classification', {}).get('type', '')
        agents = state['routing_info'].get('agents', [])

        if qtype == 'hybrid' and 'vector' in agents and not state.get('vector_result'):
            if self.enable_vector:
                return "vector_agent"
        return "synthesize"

    # ── SQL agent ────────────────────────────────────────────────────────

    def _execute_sql_agent(self, state: AgentState) -> AgentState:
        logger.info("Executing SQL Agent")
        try:
            if 'sql' in self.agents:
                query = state['query']
                if state['routing_info']['classification']['type'] == 'hybrid':
                    query = self._extract_sql_query(state['query'])
                    logger.info(f"Extracted SQL query: {query}")

                result = self.agents['sql'].process_query(
                    query,
                    conversation_context=state.get("conversation_context"),
                )
                state['sql_result'] = result
                logger.info("SQL Agent execution complete")
            else:
                state['sql_result'] = {
                    'success': False,
                    'error': 'SQL Agent not available'
                }
        except Exception as e:
            logger.error(f"SQL Agent error: {e}")
            state['sql_result'] = {
                'success': False,
                'error': str(e)
            }
        return state

    def _check_if_hybrid(self, state: AgentState) -> str:
        agents = state['routing_info'].get('agents', [])
        if 'vector' in agents and not state.get('vector_result'):
            return "vector_agent"
        return "synthesize"

    # ── Vector agent ─────────────────────────────────────────────────────

    def _execute_vector_agent(self, state: AgentState) -> AgentState:
        logger.info("Executing Vector Agent")
        try:
            if 'vector' in self.agents:
                query = state['query']
                qtype = state['routing_info']['classification']['type']
                if qtype == 'hybrid':
                    query = self._extract_vector_query_local(state['query'])
                    logger.info(f"Extracted Vector query (local): {query}")

                limit = max(1, min(int(state.get("max_results") or 10), 100))
                filters: Dict[str, Any] = {}
                explicit = state.get("lang_filter")
                if explicit:
                    filters["lang"] = explicit
                else:
                    hinted = infer_lang_filter_from_query(state["query"])
                    if hinted:
                        filters["lang"] = hinted
                        logger.info(f"Vector search: inferred language filter lang={hinted}")

                result = self.agents["vector"].process_query(
                    query,
                    limit=limit,
                    filters=filters,
                    conversation_context=state.get("conversation_context"),
                )
                state['vector_result'] = result
                logger.info("Vector Agent execution complete")
            else:
                state['vector_result'] = {
                    'success': False,
                    'error': 'Vector Agent not available'
                }
        except Exception as e:
            logger.error(f"Vector Agent error: {e}")
            state['vector_result'] = {
                'success': False,
                'error': str(e)
            }
        return state

    # ── query splitting (local, no LLM) ──────────────────────────────────

    def _extract_sql_query(self, query: str) -> str:
        """Extract SQL-relevant part using local heuristics."""
        for connector in [' and also ', ' also ', ' and suggest ', ' and find ', ' plus ', ' as well as ']:
            if connector in query.lower():
                parts = query.lower().split(connector)
                return parts[0].strip()
        return query

    def _extract_vector_query_local(self, query: str) -> str:
        """Extract semantic-relevant part using local heuristics."""
        for connector in [' and also ', ' also ', ' and suggest ', ' and find ', ' plus ', ' as well as ']:
            if connector in query.lower():
                parts = query.lower().split(connector)
                if len(parts) > 1:
                    return parts[1].strip()
        # If no explicit split, pass the whole query for semantic search
        return query

    # ── synthesis ────────────────────────────────────────────────────────

    def _synthesize_response(self, state: AgentState) -> AgentState:
        logger.info("Synthesizing final response")

        try:
            kg_result = state.get('kg_result')
            sql_result = state.get('sql_result')
            vector_result = state.get('vector_result')

            # No agent ran (UNKNOWN route or empty agents list)
            if not kg_result and not sql_result and not vector_result:
                state['final_response'] = (
                    "I couldn't determine how to answer that query. "
                    "Try asking about artists, songs, genres, moods, or lyrics — for example:\n\n"
                    "- \"Who are the top 10 most popular artists?\"\n"
                    "- \"Find songs with sad lyrics\"\n"
                    "- \"What genres does Radiohead play?\""
                )
                state['metadata'] = {
                    'agents_used': [],
                    'query_type': state.get('routing_info', {}).get('classification', {}).get('type', 'unknown'),
                    'confidence': 0.0,
                }
                return state

            # KG-only: instant answer
            if kg_result and kg_result.get('success') and not vector_result and not sql_result:
                state['final_response'] = kg_result['data']['answer']

            # KG + Vector (hybrid): template merge, no LLM
            elif kg_result and kg_result.get('success') and vector_result and vector_result.get('success'):
                state['final_response'] = self._merge_kg_vector(
                    state['query'],
                    kg_result['data']['answer'],
                    vector_result['data']['answer'],
                )

            # SQL-only
            elif sql_result and not vector_result:
                if sql_result.get('success'):
                    state['final_response'] = sql_result['data']['answer']
                else:
                    state['final_response'] = f"Error: {sql_result.get('error', 'Unknown error')}"

            # Vector-only
            elif vector_result and not sql_result and not (kg_result and kg_result.get('success')):
                if vector_result.get('success'):
                    state['final_response'] = vector_result['data']['answer']
                else:
                    state['final_response'] = f"Error: {vector_result.get('error', 'Unknown error')}"

            # SQL + Vector (old hybrid path, LLM synthesis)
            elif sql_result and vector_result:
                state['final_response'] = self._synthesize_hybrid_response(
                    state['query'],
                    sql_result,
                    vector_result,
                    conversation_context=state.get("conversation_context"),
                )

            # KG failed, fall back to SQL agent
            elif kg_result and not kg_result.get('success') and not sql_result and not vector_result:
                logger.warning("KG failed and no other agent ran -- attempting SQL fallback")
                if 'sql' in self.agents:
                    fb = self.agents['sql'].process_query(
                        state['query'],
                        conversation_context=state.get("conversation_context"),
                    )
                    if fb.get('success'):
                        state['final_response'] = fb['data']['answer']
                        state['sql_result'] = fb
                    else:
                        state['final_response'] = f"Error: {fb.get('error', 'Unknown error')}"
                else:
                    state['final_response'] = "I couldn't process your query. Please try rephrasing."
            else:
                state['final_response'] = "I couldn't process your query. Please try rephrasing."

            # Metadata
            agents_used = []
            if kg_result and kg_result.get('success'):
                agents_used.append('kg_direct')
            if sql_result and sql_result.get('success'):
                agents_used.append('sql')
            if vector_result and vector_result.get('success'):
                agents_used.append('vector')

            state['metadata'] = {
                'agents_used': agents_used or state['routing_info'].get('agents', []),
                'query_type': state['routing_info']['classification']['type'],
                'confidence': state['routing_info']['classification']['confidence'],
            }
            vf = state.get("vector_result", {}) or {}
            vmeta = vf.get("metadata") or {}
            if vmeta.get("filters"):
                state["metadata"]["vector_filters"] = vmeta["filters"]

            logger.info("Response synthesis complete")

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            state['final_response'] = f"Error synthesizing response: {str(e)}"

        return state

    def _merge_kg_vector(self, query: str, kg_answer: str, vector_answer: str) -> str:
        """Merge KG + Vector results using a template (no LLM)."""
        lines = [
            "## Structured Analysis\n",
            kg_answer,
            "\n\n## Semantic Search Results\n",
            vector_answer,
        ]
        return "\n".join(lines)

    def _synthesize_hybrid_response(
        self,
        query: str,
        sql_result: Dict,
        vector_result: Dict,
        conversation_context: Optional[str] = None,
    ) -> str:
        try:
            sql_answer = sql_result.get('data', {}).get('answer', 'No SQL result')
            vector_answer = vector_result.get('data', {}).get('answer', 'No vector result')

            prior = ""
            if conversation_context and conversation_context.strip():
                prior = f"""
Prior conversation (for resolving follow-ups; do not invent facts not in SQL/vector results):
{conversation_context.strip()}

"""

            prompt = f"""You are a music information retrieval assistant for Music Maven, powered by the Music4All dataset.
Synthesize the following results into a coherent, helpful response using PROPER MARKDOWN FORMATTING.
{prior}
User Query: {query}

Structured Data Analysis (SQL):
{sql_answer}

Semantic Search Results (Vector):
{vector_answer}

IMPORTANT RULES:
- Only include song titles, artists, and attributes that appear in the results above.
- Do NOT invent views, streams, play counts, or any statistics not in the data.
- Do NOT add descriptions or context beyond what the data provides.

Format your response using proper markdown:
1. Use a brief introductory sentence
2. List songs using markdown bullet points (- **"Song Title"** - Artist - key attributes from the data)
3. Add a "## Key Insights" section combining patterns from both SQL and semantic results
4. Use **bold** for song titles, artist names, and important attributes
5. Use proper markdown headings (##) for sections
6. Keep the response concise and well-structured

Response:"""

            response = self.llm.invoke(prompt)
            return response.content

        except Exception as e:
            logger.error(f"Hybrid synthesis error: {e}")
            return f"## SQL Analysis\n{sql_answer}\n\n## Semantic Search\n{vector_answer}"

    # ── public API ───────────────────────────────────────────────────────

    def process_query(
        self,
        query: str,
        max_results: int = 10,
        lang_filter: Optional[str] = None,
        use_kg: bool = True,
        conversation_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info(f"Processing query through orchestrator: {query}")

        initial_state: AgentState = {
            'query': query,
            'routing_info': {},
            'kg_result': None,
            'sql_result': None,
            'vector_result': None,
            'final_response': None,
            'error': None,
            'metadata': {},
            'max_results': max(1, min(int(max_results), 100)),
            'lang_filter': lang_filter,
            'use_kg': bool(use_kg),
            'conversation_context': (conversation_context or None),
        }

        try:
            final_state = self.graph.invoke(initial_state)

            response = {
                'query': query,
                'answer': final_state.get('final_response') or 'No response generated',
                'metadata': final_state.get('metadata', {}),
                'routing': final_state.get('routing_info', {}),
                'success': final_state.get('final_response') is not None,
            }

            if final_state.get('kg_result'):
                response['kg_result'] = final_state['kg_result']
            if final_state.get('sql_result'):
                response['sql_result'] = final_state['sql_result']
            if final_state.get('vector_result'):
                response['vector_result'] = final_state['vector_result']
                vr = final_state['vector_result']
                if vr.get('success') and vr.get('data', {}).get('results') is not None:
                    response['results'] = vr['data']['results']

            logger.info("Query processing complete")
            return response

        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            return {
                'query': query,
                'answer': f"An error occurred: {str(e)}",
                'metadata': {},
                'success': False,
                'error': str(e),
            }

    def get_agent_info(self) -> Dict[str, Any]:
        info = {
            'orchestrator': 'LangGraph Multi-Agent System (KG-enhanced)',
            'agents': {},
        }

        if self.kg_engine is not None:
            info['agents']['kg'] = {
                'name': 'KGAgent',
                'type': 'knowledge_graph',
                'description': 'Answers factual queries from pre-built Knowledge Graph (zero LLM calls)',
            }

        for name, agent in self.agents.items():
            info['agents'][name] = agent.get_capabilities()

        return info
