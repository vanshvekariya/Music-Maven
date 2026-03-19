"""Main application entry point for Music Maven Multi-Agent System"""

import sys
from typing import Optional
from pathlib import Path

from loguru import logger

from .agents.orchestrator import MultiAgentOrchestrator
from .config.settings import get_settings


class MusicMavenApp:
    """
    Main application class for Music Maven.
    Provides a high-level interface to the multi-agent system.

    Phase 3: KG-enhanced mode.  The Knowledge Graph is loaded at startup
    and used as a fast path for factual / analytical queries (zero LLM calls).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        enable_sql: bool = True,
        enable_vector: bool = True,
    ):
        self.settings = get_settings()
        self._configure_logging()

        logger.info("Initializing Music Maven Multi-Agent System")

        # Load Knowledge Graph
        kg_engine = None
        if self.settings.kg_enable:
            kg_engine = self._load_kg()

        # Initialize orchestrator
        try:
            self.orchestrator = MultiAgentOrchestrator(
                api_key=api_key,
                model=model,
                enable_sql=enable_sql,
                enable_vector=enable_vector,
                kg_engine=kg_engine,
            )
            logger.info("Application initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            raise

    def _load_kg(self):
        """Load the Knowledge Graph and return a KGQueryEngine instance."""
        try:
            from .knowledge_graph.kg_builder import KnowledgeGraphBuilder
            from .knowledge_graph.kg_query_engine import KGQueryEngine

            G = KnowledgeGraphBuilder.load_or_build()
            engine = KGQueryEngine(G)
            logger.info("Knowledge Graph loaded and query engine ready")
            return engine
        except Exception as e:
            logger.warning(f"Failed to load Knowledge Graph: {e}")
            return None

    def _configure_logging(self) -> None:
        """Configure logging settings"""
        logger.remove()

        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            level=self.settings.log_level,
        )

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        logger.add(
            log_dir / "music_maven_{time}.log",
            rotation="100 MB",
            retention="10 days",
            level="DEBUG",
        )

    def query(self, query: str) -> dict:
        logger.info(f"Processing query: {query}")
        try:
            response = self.orchestrator.process_query(query)
            return response
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                'query': query,
                'answer': f"An error occurred: {str(e)}",
                'success': False,
                'error': str(e),
            }

    def get_system_info(self) -> dict:
        return self.orchestrator.get_agent_info()

    def interactive_mode(self) -> None:
        """Run the application in interactive CLI mode."""
        print("\n" + "=" * 70)
        print("  Music Maven - Music Information Retrieval System")
        print("=" * 70)
        print("\nWelcome! Ask questions about the Music4All dataset.")
        print("Type 'help' for examples, 'info' for system info, or 'quit' to exit.\n")

        while True:
            try:
                user_input = input("\nYour question: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for using Music Maven! Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'info':
                    self._show_system_info()
                    continue
                elif user_input.lower() == 'clear':
                    print("\n" * 50)
                    continue

                print("\nProcessing your query...")
                response = self.query(user_input)

                print("\n" + "-" * 70)
                print("ANSWER:")
                print("-" * 70)
                print(response.get('answer', 'No answer generated'))

                if response.get('metadata'):
                    metadata = response['metadata']
                    print(f"\nQuery Type: {metadata.get('query_type', 'unknown')}")
                    print(f"Agents Used: {', '.join(metadata.get('agents_used', []))}")
                    print(f"Confidence: {metadata.get('confidence', 0):.2%}")

                print("-" * 70)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit or continue asking questions.")
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"\nError: {str(e)}")

    def _show_help(self) -> None:
        print("\n" + "=" * 70)
        print("  HELP - Example Queries")
        print("=" * 70)
        print("\nKG-Direct Queries (instant, 0 LLM calls):")
        print("  - Who are the top 10 most popular artists?")
        print("  - How many songs are in English?")
        print("  - What are the most common genres?")
        print("  - Songs by Queen")
        print("  - Compare Queen and The Beatles")
        print("  - Stats for David Bowie")
        print("  - Average tempo of rock songs")

        print("\nSQL Queries (complex analysis):")
        print("  - Songs with energy above 0.8 and danceability above 0.7")
        print("  - Songs with tempo between 120 and 140 BPM that have lyrics")

        print("\nVector/Semantic Queries:")
        print("  - Find songs that feel melancholic")
        print("  - Chill lo-fi hip hop")
        print("  - Songs about heartbreak")

        print("\nHybrid Queries:")
        print("  - Popular sad rock songs")
        print("  - High energy songs in English")

        print("\nCommands:")
        print("  - help  - Show this help")
        print("  - info  - Show system info")
        print("  - clear - Clear screen")
        print("  - quit  - Exit")
        print("=" * 70)

    def _show_system_info(self) -> None:
        info = self.get_system_info()
        print("\n" + "=" * 70)
        print("  SYSTEM INFORMATION")
        print("=" * 70)
        print(f"\nOrchestrator: {info.get('orchestrator', 'Unknown')}")

        print("\nAvailable Agents:")
        for agent_name, agent_info in info.get('agents', {}).items():
            print(f"\n  - {agent_info.get('name', agent_name)}")
            print(f"    Type: {agent_info.get('type', 'unknown')}")
            print(f"    Description: {agent_info.get('description', 'N/A')}")

        print(f"\nConfiguration:")
        print(f"  LLM Model: {self.settings.llm_model}")
        print(f"  SQL Database: {self.settings.sql_db_path}")
        print(f"  Vector DB: Qdrant ({self.settings.qdrant_host}:{self.settings.qdrant_port})")
        print(f"  Embedding Model: {self.settings.local_embedding_model}")
        print(f"  KG Enabled: {self.settings.kg_enable}")
        print("=" * 70)


def main():
    """Main entry point for CLI application"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Music Maven - Music Information Retrieval System'
    )
    parser.add_argument('--query', '-q', type=str, help='Single query to process')
    parser.add_argument('--api-key', type=str, help='OpenAI/OpenRouter API key')
    parser.add_argument('--model', type=str, help='LLM model to use')
    parser.add_argument('--no-sql', action='store_true', help='Disable SQL agent')
    parser.add_argument('--no-vector', action='store_true', help='Disable Vector agent')
    parser.add_argument('--info', action='store_true', help='Show system info and exit')

    args = parser.parse_args()

    try:
        app = MusicMavenApp(
            api_key=args.api_key,
            model=args.model,
            enable_sql=not args.no_sql,
            enable_vector=not args.no_vector,
        )

        if args.info:
            app._show_system_info()
            return

        if args.query:
            response = app.query(args.query)
            print("\n" + "=" * 70)
            print("ANSWER:")
            print("=" * 70)
            print(response.get('answer', 'No answer generated'))
            print("=" * 70)
            return

        app.interactive_mode()

    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
