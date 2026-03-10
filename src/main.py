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

    Phase 2: Vector + SQL mode (enable_vector=True).
    Requires Qdrant to be running and both collections indexed via the embedding pipeline.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        enable_sql: bool = True,
        enable_vector: bool = True
    ):
        """
        Initialize the application.
        
        Args:
            api_key: OpenAI/OpenRouter API key
            model: LLM model to use
            enable_sql: Enable SQL agent
            enable_vector: Enable Vector agent (requires Qdrant + indexed collections)
        """
        self.settings = get_settings()
        
        # Configure logging
        self._configure_logging()
        
        logger.info("Initializing Music Maven Multi-Agent System")
        
        # Initialize orchestrator
        try:
            self.orchestrator = MultiAgentOrchestrator(
                api_key=api_key,
                model=model,
                enable_sql=enable_sql,
                enable_vector=enable_vector
            )
            logger.info("Application initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            raise
    
    def _configure_logging(self) -> None:
        """Configure logging settings"""
        logger.remove()  # Remove default handler
        
        # Add console handler
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            level=self.settings.log_level
        )
        
        # Add file handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger.add(
            log_dir / "music_maven_{time}.log",
            rotation="100 MB",
            retention="10 days",
            level="DEBUG"
        )
    
    def query(self, query: str) -> dict:
        """
        Process a user query.
        
        Args:
            query: Natural language query
            
        Returns:
            Response dictionary
        """
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
                'error': str(e)
            }
    
    def get_system_info(self) -> dict:
        """
        Get information about the system and available agents.
        
        Returns:
            System information dictionary
        """
        return self.orchestrator.get_agent_info()
    
    def interactive_mode(self) -> None:
        """
        Run the application in interactive CLI mode.
        """
        print("\n" + "="*70)
        print("  Music Maven — Music Information Retrieval System")
        print("="*70)
        print("\nWelcome! Ask questions about the Music4All dataset.")
        print("Type 'help' for examples, 'info' for system info, or 'quit' to exit.\n")
        
        while True:
            try:
                # Get user input
                user_input = input("\n🔍 Your question: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
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
                
                # Process query
                print("\n⏳ Processing your query...")
                response = self.query(user_input)
                
                # Display results
                print("\n" + "-"*70)
                print("📊 ANSWER:")
                print("-"*70)
                print(response.get('answer', 'No answer generated'))
                
                # Show metadata if available
                if response.get('metadata'):
                    metadata = response['metadata']
                    print(f"\n📌 Query Type: {metadata.get('query_type', 'unknown')}")
                    print(f"🤖 Agents Used: {', '.join(metadata.get('agents_used', []))}")
                    print(f"✅ Confidence: {metadata.get('confidence', 0):.2%}")
                
                print("-"*70)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit or continue asking questions.")
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"\n❌ Error: {str(e)}")
    
    def _show_help(self) -> None:
        """Display help information"""
        print("\n" + "="*70)
        print("  HELP - Example Queries")
        print("="*70)
        print("\n📊 SQL/Analytical Queries (structured data):")
        print("  • Who are the top 10 most popular artists?")
        print("  • How many songs are in English?")
        print("  • What are the most common genres?")
        print("  • Find high-energy danceable songs")
        print("  • Songs with tempo between 120 and 140 BPM that have lyrics")

        print("\n🔍 Vector/Semantic Queries (coming in Phase 2):")
        print("  • Find songs that feel melancholic")
        print("  • Songs lyrically similar to Bohemian Rhapsody")
        print("  • Music about lost love")

        print("\n🔄 Hybrid Queries (coming in Phase 2):")
        print("  • Popular sad songs by Queen")
        print("  • High energy rock songs in English")
        
        print("\n💡 Commands:")
        print("  • help  - Show this help message")
        print("  • info  - Show system information")
        print("  • clear - Clear screen")
        print("  • quit  - Exit the application")
        print("="*70)
    
    def _show_system_info(self) -> None:
        """Display system information"""
        info = self.get_system_info()
        
        print("\n" + "="*70)
        print("  SYSTEM INFORMATION")
        print("="*70)
        print(f"\n🤖 Orchestrator: {info.get('orchestrator', 'Unknown')}")
        
        print("\n📦 Available Agents:")
        for agent_name, agent_info in info.get('agents', {}).items():
            print(f"\n  • {agent_info.get('name', agent_name)}")
            print(f"    Type: {agent_info.get('type', 'unknown')}")
            print(f"    Description: {agent_info.get('description', 'N/A')}")
        
        print("\n⚙️  Configuration:")
        print(f"  • LLM Model: {self.settings.llm_model}")
        print(f"  • SQL Database: {self.settings.sql_db_path}")
        print(f"  • Vector DB: Qdrant ({self.settings.qdrant_host}:{self.settings.qdrant_port})")
        print(f"  • Embedding Model: {self.settings.local_embedding_model}")
        print("="*70)


def main():
    """Main entry point for CLI application"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Music Maven — Music Information Retrieval System'
    )
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Single query to process (non-interactive mode)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenAI/OpenRouter API key (or set OPENAI_API_KEY env var)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='LLM model to use'
    )
    parser.add_argument(
        '--no-sql',
        action='store_true',
        help='Disable SQL agent'
    )
    parser.add_argument(
        '--no-vector',
        action='store_true',
        help='Disable Vector agent'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show system information and exit'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize application
        app = MusicMavenApp(
            api_key=args.api_key,
            model=args.model,
            enable_sql=not args.no_sql,
            enable_vector=not args.no_vector
        )
        
        # Show info and exit
        if args.info:
            app._show_system_info()
            return
        
        # Single query mode
        if args.query:
            response = app.query(args.query)
            print("\n" + "="*70)
            print("ANSWER:")
            print("="*70)
            print(response.get('answer', 'No answer generated'))
            print("="*70)
            return
        
        # Interactive mode
        app.interactive_mode()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
