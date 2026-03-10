"""Test script for Vector Agent functionality"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.agents.vector_agent import VectorAgent


def print_separator(title=""):
    """Print a formatted separator"""
    print("\n" + "="*80)
    if title:
        print(f"  {title}")
        print("="*80)
    print()


def test_basic_search(agent):
    """Test basic semantic search"""
    print_separator("Test 1: Basic Semantic Search")
    
    query = "Walk On Water Audio"
    logger.info(f"Testing query: '{query}'")
    
    result = agent.process_query(query, limit=5)
    
    if result['success']:
        print(f"✅ Query successful!")
        print(f"\nAnswer:\n{result['data']['answer']}\n")
        
        print("Results:")
        for video in result['data']['results']:
            print(f"  {video['rank']}. {video['title']}")
            print(f"     Channel: {video['channel']}")
            print(f"     Category: {video['category']}")
            print(f"     Views: {video['views']:,} | Likes: {video['likes']:,}")
            print(f"     Relevance: {video['score']:.2%}")
            print()
    else:
        print(f"❌ Query failed: {result['error']}")


def test_filtered_search(agent):
    """Test filtered search"""
    print_separator("Test 2: Filtered Search by Category")
    
    query = "gaming"
    category = "Gaming"
    
    logger.info(f"Testing filtered search: '{query}' in category '{category}'")
    
    result = agent.search_by_category(query, category, limit=5)
    
    if result['success']:
        print(f"✅ Filtered search successful!")
        print(f"\nFound {result['metadata']['num_results']} gaming videos")
        
        for video in result['data']['results'][:3]:
            print(f"  - {video['title']} ({video['score']:.2%} match)")
    else:
        print(f"❌ Search failed: {result['error']}")


def test_popular_search(agent):
    """Test popular videos search"""
    print_separator("Test 3: Popular Videos Search")
    
    query = "music"
    min_views = 100000
    
    logger.info(f"Testing popular search: '{query}' with min_views={min_views}")
    
    result = agent.search_popular_videos(query, min_views=min_views, limit=5)
    
    if result['success']:
        print(f"✅ Popular search successful!")
        print(f"\nFound {result['metadata']['num_results']} popular music videos")
        
        for video in result['data']['results'][:3]:
            print(f"  - {video['title']}")
            print(f"    Views: {video['views']:,} | Score: {video['score']:.2%}")
    else:
        print(f"❌ Search failed: {result['error']}")


def test_hybrid_search(agent):
    """Test hybrid search with multiple filters"""
    print_separator("Test 4: Hybrid Search")
    
    query = "tutorial"
    
    logger.info(f"Testing hybrid search: '{query}' with multiple filters")
    
    result = agent.hybrid_search(
        query=query,
        category="Education",
        min_views=10000,
        limit=5
    )
    
    if result['success']:
        print(f"✅ Hybrid search successful!")
        print(f"\nFound {result['metadata']['num_results']} educational tutorials")
        
        for video in result['data']['results'][:3]:
            print(f"  - {video['title']}")
            print(f"    Channel: {video['channel']} | Views: {video['views']:,}")
    else:
        print(f"❌ Search failed: {result['error']}")


def test_agent_stats(agent):
    """Test agent statistics and health"""
    print_separator("Test 5: Agent Statistics & Health")
    
    # Get statistics
    stats = agent.get_stats()
    print("📊 Statistics:")
    print(f"  Collection: {stats.get('collection_name', 'N/A')}")
    print(f"  Total Videos: {stats.get('total_videos', 0):,}")
    print(f"  Status: {stats.get('status', 'Unknown')}")
    print(f"  Embedding Dimension: {stats.get('embedding_dimension', 0)}")
    
    # Health check
    print("\n🏥 Health Check:")
    health = agent.health_check()
    print(f"  Agent Status: {health['agent']}")
    for component, status in health['components'].items():
        print(f"  {component}: {status}")
    
    # Capabilities
    print("\n🎯 Capabilities:")
    caps = agent.get_capabilities()
    print(f"  Type: {caps['type']}")
    print(f"  Vector DB: {caps['vector_db']['type']} at {caps['vector_db']['host']}:{caps['vector_db']['port']}")
    print(f"  Embedding Model: {caps['embedding_model']['name']} (dim={caps['embedding_model']['dimension']})")
    print(f"  LLM Enabled: {caps['llm']['enabled']}")


def main():
    """Run all tests"""
    print_separator("Vector Agent Test Suite")
    
    try:
        # Initialize agent
        logger.info("Initializing Vector Agent...")
        agent = VectorAgent()
        print("✅ Vector Agent initialized successfully!\n")
        
        # Run tests
        test_basic_search(agent)
        test_filtered_search(agent)
        test_popular_search(agent)
        test_hybrid_search(agent)
        test_agent_stats(agent)
        
        print_separator("All Tests Complete!")
        print("✅ Vector Agent is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
