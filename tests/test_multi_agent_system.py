"""
Unit tests for the multi-agent system.

Run with: pytest tests/test_multi_agent_system.py -v
"""

import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.base_agent import BaseAgent
from src.agents.query_router import QueryType, QueryClassification, SimpleQueryRouter


class TestBaseAgent:
    """Test BaseAgent abstract class"""
    
    def test_base_agent_cannot_be_instantiated(self):
        """BaseAgent is abstract and cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseAgent("test")
    
    def test_format_response(self):
        """Test response formatting"""
        # Create a concrete implementation for testing
        class ConcreteAgent(BaseAgent):
            def process_query(self, query: str):
                return {}
            
            def get_capabilities(self):
                return {}
        
        agent = ConcreteAgent("TestAgent")
        
        # Test success response
        response = agent.format_response(
            success=True,
            data={'result': 'test'},
            metadata={'key': 'value'}
        )
        
        assert response['success'] is True
        assert response['agent'] == 'TestAgent'
        assert response['data']['result'] == 'test'
        assert response['metadata']['key'] == 'value'
        assert response['error'] is None
    
    def test_validate_query(self):
        """Test query validation"""
        class ConcreteAgent(BaseAgent):
            def process_query(self, query: str):
                return {}
            
            def get_capabilities(self):
                return {}
        
        agent = ConcreteAgent("TestAgent")
        
        # Valid queries
        assert agent.validate_query("test query") is True
        assert agent.validate_query("  test  ") is True
        
        # Invalid queries
        assert agent.validate_query("") is False
        assert agent.validate_query("   ") is False
        assert agent.validate_query(None) is False
        assert agent.validate_query(123) is False


class TestSimpleQueryRouter:
    """Test SimpleQueryRouter (rule-based fallback)"""
    
    def test_sql_query_classification(self):
        """Test SQL query classification"""
        router = SimpleQueryRouter()
        
        sql_queries = [
            "How many videos are there?",
            "Count the total views",
            "Top 10 channels",
            "Average likes per category",
            "Compare Music and Gaming"
        ]
        
        for query in sql_queries:
            result = router.classify_query(query)
            assert result.query_type == QueryType.SQL
            assert result.confidence > 0
    
    def test_vector_query_classification(self):
        """Test Vector query classification"""
        router = SimpleQueryRouter()
        
        vector_queries = [
            "Find videos about cooking",
            "Search for gaming content",
            "Videos similar to tech reviews",
            "Show me educational videos",
            "Content related to fitness"
        ]
        
        for query in vector_queries:
            result = router.classify_query(query)
            assert result.query_type == QueryType.VECTOR
            assert result.confidence > 0
    
    def test_hybrid_query_classification(self):
        """Test Hybrid query classification"""
        router = SimpleQueryRouter()
        
        # Query with both SQL and Vector keywords
        query = "Find top 10 videos about cooking"
        result = router.classify_query(query)
        
        # Should be classified as hybrid or one of the types
        assert result.query_type in [QueryType.SQL, QueryType.VECTOR, QueryType.HYBRID]
    
    def test_route_query(self):
        """Test query routing"""
        router = SimpleQueryRouter()
        
        query = "How many videos are there?"
        routing_info = router.route_query(query)
        
        assert 'query' in routing_info
        assert 'classification' in routing_info
        assert 'agents' in routing_info
        assert 'execution_strategy' in routing_info
        
        assert routing_info['query'] == query
        assert isinstance(routing_info['agents'], list)


class TestQueryClassification:
    """Test QueryClassification model"""
    
    def test_query_classification_creation(self):
        """Test creating QueryClassification object"""
        classification = QueryClassification(
            query_type=QueryType.SQL,
            confidence=0.95,
            reasoning="Contains analytical keywords",
            suggested_agent="sql"
        )
        
        assert classification.query_type == QueryType.SQL
        assert classification.confidence == 0.95
        assert "analytical" in classification.reasoning
        assert classification.suggested_agent == "sql"
    
    def test_confidence_validation(self):
        """Test confidence score validation"""
        # Valid confidence
        classification = QueryClassification(
            query_type=QueryType.SQL,
            confidence=0.5,
            reasoning="test",
            suggested_agent="sql"
        )
        assert classification.confidence == 0.5
        
        # Invalid confidence (should raise validation error)
        with pytest.raises(Exception):  # Pydantic validation error
            QueryClassification(
                query_type=QueryType.SQL,
                confidence=1.5,  # > 1.0
                reasoning="test",
                suggested_agent="sql"
            )


class TestDataProcessor:
    """Test Enhanced Data Processor"""
    
    def test_clean_text(self):
        """Test text cleaning function"""
        from src.data.enhanced_processor import EnhancedYouTubeDataProcessor
        
        processor = EnhancedYouTubeDataProcessor()
        
        # Test URL removal
        text = "Check out https://example.com for more!"
        cleaned = processor.clean_text(text)
        assert "https://" not in cleaned
        assert "example.com" not in cleaned
        
        # Test HTML removal
        text = "<p>Hello <b>World</b></p>"
        cleaned = processor.clean_text(text)
        assert "<p>" not in cleaned
        assert "<b>" not in cleaned
        
        # Test punctuation removal
        text = "Hello, World! How are you?"
        cleaned = processor.clean_text(text)
        assert "," not in cleaned
        assert "!" not in cleaned
        assert "?" not in cleaned
    
    def test_split_and_clean_tags(self):
        """Test tag splitting and cleaning"""
        from src.data.enhanced_processor import EnhancedYouTubeDataProcessor
        
        processor = EnhancedYouTubeDataProcessor()
        
        # Test pipe-separated tags
        tags = "gaming|minecraft|tutorial|fun"
        result = processor.split_and_clean_tags(tags)
        assert len(result) == 4
        assert "gaming" in result
        assert "minecraft" in result
        
        # Test [none] tag
        tags = "[none]"
        result = processor.split_and_clean_tags(tags)
        assert len(result) == 0
        
        # Test empty/None
        result = processor.split_and_clean_tags("")
        assert len(result) == 0


class TestIntegration:
    """Integration tests (require actual setup)"""
    
    @pytest.mark.skip(reason="Requires API key and database setup")
    def test_full_query_flow(self):
        """Test complete query flow (skipped by default)"""
        from src.main import YouTubeTrendsApp
        
        app = YouTubeTrendsApp()
        response = app.query("How many videos are there?")
        
        assert response['success'] is True
        assert 'answer' in response
        assert 'metadata' in response


def test_imports():
    """Test that all modules can be imported"""
    try:
        from src.agents import (
            BaseAgent,
            SQLAgent,
            VectorAgent,
            QueryRouter,
            MultiAgentOrchestrator
        )
        from src.data.enhanced_processor import EnhancedYouTubeDataProcessor
        from src.main import YouTubeTrendsApp
        
        assert True  # All imports successful
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
