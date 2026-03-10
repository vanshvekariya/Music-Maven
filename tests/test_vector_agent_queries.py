"""
Sample Queries for Testing Vector Database Agent

This module contains comprehensive test queries designed to test all features
of the Vector Agent including semantic search, filtering, recommendations,
and hybrid search capabilities.

Run this file to test the vector agent with various query types.
"""

from typing import List, Dict, Any
from loguru import logger
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.vector_agent import VectorAgent


# ==================== SAMPLE QUERIES BY FEATURE ====================

SEMANTIC_SEARCH_QUERIES = [
    # Basic semantic search
    "funny cat videos",
    "cooking tutorials for beginners",
    "machine learning explained",
    "motivational speeches",
    "travel vlogs in Europe",
    
    # Topic-based search
    "videos about artificial intelligence",
    "content related to fitness and workout",
    "gaming highlights and montages",
    "music covers and performances",
    "educational content about space",
    
    # Conceptual queries
    "how to start a business",
    "climate change documentaries",
    "meditation and mindfulness",
    "DIY home improvement projects",
    "cryptocurrency explained",
    
    # Natural language queries
    "show me videos discussing mental health",
    "find content about sustainable living",
    "videos explaining quantum physics",
    "comedy sketches and funny moments",
    "tutorials on digital art and drawing"
]

CATEGORY_FILTERED_QUERIES = [
    # Gaming category
    {
        "query": "competitive gameplay and strategies",
        "category": "Gaming",
        "description": "Gaming videos about competitive play"
    },
    {
        "query": "game reviews and walkthroughs",
        "category": "Gaming",
        "description": "Gaming content reviews"
    },
    
    # Music category
    {
        "query": "acoustic guitar covers",
        "category": "Music",
        "description": "Music videos with guitar"
    },
    {
        "query": "electronic dance music",
        "category": "Music",
        "description": "EDM music content"
    },
    
    # Education category
    {
        "query": "programming tutorials",
        "category": "Education",
        "description": "Educational programming content"
    },
    {
        "query": "science experiments",
        "category": "Education",
        "description": "Educational science videos"
    },
    
    # Entertainment category
    {
        "query": "movie reviews and analysis",
        "category": "Entertainment",
        "description": "Entertainment content about movies"
    },
    {
        "query": "celebrity interviews",
        "category": "Entertainment",
        "description": "Entertainment interviews"
    },
    
    # Science & Technology
    {
        "query": "latest tech gadgets",
        "category": "Science & Technology",
        "description": "Tech reviews and news"
    },
    {
        "query": "smartphone comparisons",
        "category": "Science & Technology",
        "description": "Tech comparison videos"
    }
]

POPULAR_VIDEO_QUERIES = [
    # High view count searches
    {
        "query": "viral challenges",
        "min_views": 1000000,
        "description": "Viral videos with 1M+ views"
    },
    {
        "query": "trending music videos",
        "min_views": 5000000,
        "description": "Popular music with 5M+ views"
    },
    {
        "query": "gaming highlights",
        "min_views": 500000,
        "description": "Popular gaming content"
    },
    {
        "query": "cooking recipes",
        "min_views": 100000,
        "description": "Popular cooking videos"
    },
    {
        "query": "tech reviews",
        "min_views": 250000,
        "description": "Popular tech content"
    }
]

HYBRID_SEARCH_QUERIES = [
    # Semantic + Category
    {
        "query": "beginner tutorials",
        "category": "Education",
        "min_views": 50000,
        "description": "Educational tutorials with good engagement"
    },
    {
        "query": "funny moments compilation",
        "category": "Entertainment",
        "min_views": 100000,
        "max_views": 1000000,
        "description": "Moderately popular entertainment content"
    },
    {
        "query": "workout routines",
        "category": "Howto & Style",
        "min_views": 10000,
        "description": "Fitness content with minimum engagement"
    },
    {
        "query": "game trailers",
        "category": "Gaming",
        "min_views": 500000,
        "description": "Popular gaming trailers"
    },
    {
        "query": "music production tutorials",
        "category": "Music",
        "min_views": 25000,
        "max_views": 500000,
        "description": "Music education with specific view range"
    }
]

SIMILARITY_SEARCH_QUERIES = [
    # These would need actual video IDs from your database
    # Format: {"video_id": "actual_id", "description": "what to expect"}
    {
        "video_id": "placeholder_id_1",
        "description": "Find videos similar to a gaming video"
    },
    {
        "video_id": "placeholder_id_2",
        "description": "Find videos similar to a cooking tutorial"
    },
    {
        "video_id": "placeholder_id_3",
        "description": "Find videos similar to a music video"
    }
]

ADVANCED_QUERIES = [
    # Complex semantic queries
    "videos explaining the difference between AI and machine learning",
    "content about sustainable fashion and ethical clothing",
    "tutorials on building web applications with React",
    "documentaries about ocean conservation",
    "videos discussing the future of renewable energy",
    
    # Multi-concept queries
    "fitness and nutrition advice for beginners",
    "travel tips and budget planning",
    "photography techniques and camera settings",
    "entrepreneurship and startup advice",
    "mental health and stress management"
]

EDGE_CASE_QUERIES = [
    # Very specific queries
    "Counter Strike Global Offensive competitive matches",
    "Italian cooking recipes with pasta",
    "Python programming for data science",
    "electric vehicle reviews and comparisons",
    "yoga for back pain relief",
    
    # Broad queries
    "videos",
    "content",
    "popular",
    
    # Queries with special characters
    "C++ programming tutorials",
    "React.js vs Vue.js comparison",
    "How to make $1000 online",
    
    # Multi-language concepts
    "anime reviews and recommendations",
    "K-pop music videos and performances",
    "Bollywood movie trailers"
]


# ==================== TEST FUNCTIONS ====================

def test_semantic_search(agent: VectorAgent):
    """Test basic semantic search functionality"""
    print("\n" + "="*80)
    print("TESTING: SEMANTIC SEARCH")
    print("="*80)
    
    for i, query in enumerate(SEMANTIC_SEARCH_QUERIES[:5], 1):  # Test first 5
        print(f"\n[{i}] Query: '{query}'")
        print("-" * 80)
        
        result = agent.process_query(query, limit=5)
        
        if result['success']:
            print(f"✓ Found {result['metadata']['num_results']} results")
            print(f"  Avg Score: {result['metadata']['avg_score']:.3f}")
            print(f"\n  Answer: {result['data']['answer'][:200]}...")
            
            print("\n  Top Results:")
            for video in result['data']['results'][:3]:
                print(f"    {video['rank']}. {video['title']}")
                print(f"       Score: {video['score']:.3f} | Views: {video['views']:,}")
        else:
            print(f"✗ Error: {result.get('error')}")


def test_category_search(agent: VectorAgent):
    """Test category-filtered search"""
    print("\n" + "="*80)
    print("TESTING: CATEGORY FILTERED SEARCH")
    print("="*80)
    
    for i, test_case in enumerate(CATEGORY_FILTERED_QUERIES[:5], 1):
        print(f"\n[{i}] {test_case['description']}")
        print(f"    Query: '{test_case['query']}'")
        print(f"    Category: {test_case['category']}")
        print("-" * 80)
        
        result = agent.search_by_category(
            query=test_case['query'],
            category=test_case['category'],
            limit=5
        )
        
        if result['success']:
            print(f"✓ Found {result['metadata']['num_results']} results in {test_case['category']}")
            
            print("\n  Top Results:")
            for video in result['data']['results'][:3]:
                print(f"    {video['rank']}. {video['title']}")
                print(f"       Category: {video['category']} | Score: {video['score']:.3f}")
        else:
            print(f"✗ Error: {result.get('error')}")


def test_popular_videos(agent: VectorAgent):
    """Test popular video search with view filters"""
    print("\n" + "="*80)
    print("TESTING: POPULAR VIDEO SEARCH")
    print("="*80)
    
    for i, test_case in enumerate(POPULAR_VIDEO_QUERIES, 1):
        print(f"\n[{i}] {test_case['description']}")
        print(f"    Query: '{test_case['query']}'")
        print(f"    Min Views: {test_case['min_views']:,}")
        print("-" * 80)
        
        result = agent.search_popular_videos(
            query=test_case['query'],
            min_views=test_case['min_views'],
            limit=5
        )
        
        if result['success']:
            print(f"✓ Found {result['metadata']['num_results']} popular videos")
            
            print("\n  Top Results:")
            for video in result['data']['results'][:3]:
                print(f"    {video['rank']}. {video['title']}")
                print(f"       Views: {video['views']:,} | Likes: {video['likes']:,}")
        else:
            print(f"✗ Error: {result.get('error')}")


def test_hybrid_search(agent: VectorAgent):
    """Test hybrid search with multiple filters"""
    print("\n" + "="*80)
    print("TESTING: HYBRID SEARCH (Semantic + Filters)")
    print("="*80)
    
    for i, test_case in enumerate(HYBRID_SEARCH_QUERIES, 1):
        print(f"\n[{i}] {test_case['description']}")
        print(f"    Query: '{test_case['query']}'")
        print(f"    Filters: Category={test_case.get('category', 'None')}, "
              f"Views={test_case.get('min_views', 0):,}-{test_case.get('max_views', '∞')}")
        print("-" * 80)
        
        result = agent.hybrid_search(
            query=test_case['query'],
            category=test_case.get('category'),
            min_views=test_case.get('min_views'),
            max_views=test_case.get('max_views'),
            limit=5
        )
        
        if result['success']:
            print(f"✓ Found {result['metadata']['num_results']} results with filters")
            
            print("\n  Top Results:")
            for video in result['data']['results'][:3]:
                print(f"    {video['rank']}. {video['title']}")
                print(f"       Category: {video['category']} | Views: {video['views']:,} | Score: {video['score']:.3f}")
        else:
            print(f"✗ Error: {result.get('error')}")


def test_advanced_queries(agent: VectorAgent):
    """Test advanced semantic queries"""
    print("\n" + "="*80)
    print("TESTING: ADVANCED SEMANTIC QUERIES")
    print("="*80)
    
    for i, query in enumerate(ADVANCED_QUERIES[:5], 1):
        print(f"\n[{i}] Query: '{query}'")
        print("-" * 80)
        
        result = agent.process_query(query, limit=5, score_threshold=0.4)
        
        if result['success']:
            print(f"✓ Found {result['metadata']['num_results']} results")
            print(f"  Avg Score: {result['metadata']['avg_score']:.3f}")
            
            print("\n  Top 2 Results:")
            for video in result['data']['results'][:2]:
                print(f"    {video['rank']}. {video['title']}")
                print(f"       Channel: {video['channel']} | Score: {video['score']:.3f}")
        else:
            print(f"✗ Error: {result.get('error')}")


def test_edge_cases(agent: VectorAgent):
    """Test edge cases and special queries"""
    print("\n" + "="*80)
    print("TESTING: EDGE CASES")
    print("="*80)
    
    for i, query in enumerate(EDGE_CASE_QUERIES[:5], 1):
        print(f"\n[{i}] Query: '{query}'")
        print("-" * 80)
        
        result = agent.process_query(query, limit=3)
        
        if result['success']:
            print(f"✓ Found {result['metadata']['num_results']} results")
            if result['data']['results']:
                print(f"  Top Result: {result['data']['results'][0]['title']}")
        else:
            print(f"✗ Error: {result.get('error')}")


def test_score_thresholds(agent: VectorAgent):
    """Test different score thresholds"""
    print("\n" + "="*80)
    print("TESTING: SCORE THRESHOLDS")
    print("="*80)
    
    query = "machine learning tutorials"
    thresholds = [0.2, 0.4, 0.6, 0.8]
    
    print(f"\nQuery: '{query}'")
    print("-" * 80)
    
    for threshold in thresholds:
        result = agent.process_query(query, limit=10, score_threshold=threshold)
        
        if result['success']:
            num_results = result['metadata']['num_results']
            avg_score = result['metadata'].get('avg_score', 0)
            print(f"  Threshold {threshold:.1f}: {num_results} results (avg score: {avg_score:.3f})")


def test_result_limits(agent: VectorAgent):
    """Test different result limits"""
    print("\n" + "="*80)
    print("TESTING: RESULT LIMITS")
    print("="*80)
    
    query = "gaming videos"
    limits = [3, 5, 10, 20]
    
    print(f"\nQuery: '{query}'")
    print("-" * 80)
    
    for limit in limits:
        result = agent.process_query(query, limit=limit)
        
        if result['success']:
            num_results = result['metadata']['num_results']
            print(f"  Limit {limit}: Retrieved {num_results} results")


def test_agent_capabilities(agent: VectorAgent):
    """Test agent utility methods"""
    print("\n" + "="*80)
    print("TESTING: AGENT CAPABILITIES & HEALTH")
    print("="*80)
    
    # Test capabilities
    print("\n1. Agent Capabilities:")
    print("-" * 80)
    caps = agent.get_capabilities()
    print(f"  Name: {caps['name']}")
    print(f"  Type: {caps['type']}")
    print(f"  Description: {caps['description']}")
    print(f"  Features: {len(caps['capabilities'])} capabilities")
    
    # Test stats
    print("\n2. Database Statistics:")
    print("-" * 80)
    stats = agent.get_stats()
    print(f"  Collection: {stats.get('collection_name', 'N/A')}")
    print(f"  Total Videos: {stats.get('total_videos', 0):,}")
    print(f"  Embedding Dimension: {stats.get('embedding_dimension', 0)}")
    
    # Test health
    print("\n3. Health Check:")
    print("-" * 80)
    health = agent.health_check()
    print(f"  Agent Status: {health['agent']}")
    for component, status in health['components'].items():
        print(f"  {component}: {status}")


def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*80)
    print("VECTOR AGENT - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    try:
        # Initialize agent
        print("\nInitializing Vector Agent...")
        agent = VectorAgent(default_limit=5, min_score_threshold=0.3)
        print("✓ Agent initialized successfully")
        
        # Run test suites
        test_agent_capabilities(agent)
        test_semantic_search(agent)
        test_category_search(agent)
        test_popular_videos(agent)
        test_hybrid_search(agent)
        test_advanced_queries(agent)
        test_edge_cases(agent)
        test_score_thresholds(agent)
        test_result_limits(agent)
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        logger.error(f"Test suite failed: {e}", exc_info=True)


def run_quick_test():
    """Run a quick test with a few queries"""
    print("\n" + "="*80)
    print("VECTOR AGENT - QUICK TEST")
    print("="*80)
    
    try:
        agent = VectorAgent()
        
        # Test 1: Simple semantic search
        print("\n1. Simple Semantic Search:")
        result = agent.process_query("funny cat videos", limit=3)
        if result['success']:
            print(f"✓ Found {result['metadata']['num_results']} results")
            print(f"  {result['data']['answer'][:150]}...")
        
        # Test 2: Category search
        print("\n2. Category Search:")
        result = agent.search_by_category("tutorials", "Education", limit=3)
        if result['success']:
            print(f"✓ Found {result['metadata']['num_results']} results in Education")
        
        # Test 3: Popular videos
        print("\n3. Popular Videos:")
        result = agent.search_popular_videos("gaming", min_views=100000, limit=3)
        if result['success']:
            print(f"✓ Found {result['metadata']['num_results']} popular gaming videos")
        
        print("\n✓ Quick test completed successfully")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Vector Agent with sample queries")
    parser.add_argument(
        "--mode",
        choices=["full", "quick"],
        default="full",
        help="Test mode: 'full' for all tests, 'quick' for basic tests"
    )
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        run_quick_test()
    else:
        run_all_tests()
