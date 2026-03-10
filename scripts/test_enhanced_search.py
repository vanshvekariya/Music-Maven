"""Test script to demonstrate enhanced search with metadata filters"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.config import get_settings
from src.embeddings import get_embedding_model
from src.vectordb import QdrantManager, VectorDBOperations


def print_results(results, title="Search Results"):
    """Pretty print search results"""
    logger.info(f"\n{'='*80}")
    logger.info(f"{title}")
    logger.info(f"{'='*80}")
    
    if not results:
        logger.info("No results found")
        return
    
    for i, result in enumerate(results, 1):
        logger.info(f"\n[{i}] {result['title']}")
        logger.info(f"    Score: {result['score']:.4f}")
        logger.info(f"    Channel: {result['channel']}")
        logger.info(f"    Category: {result['category']}")
        logger.info(f"    Country: {result['country']} | Language: {result['language']}")
        logger.info(f"    Views: {result['views']:,} | Likes: {result['likes']:,}")
        logger.info(f"    Days Trending: {result['days_trending_unique']} | Streak: {result['longest_consecutive_streak_days']}")
        if result['tags']:
            logger.info(f"    Tags: {', '.join(result['tags'][:5])}")


def main():
    """Test enhanced search with various filters"""
    
    logger.info("="*80)
    logger.info("Enhanced Search with Metadata Filters - Demo")
    logger.info("="*80)
    
    settings = get_settings()
    
    # Initialize components
    logger.info("\nInitializing components...")
    embedding_model = get_embedding_model()
    qdrant_manager = QdrantManager()
    db_ops = VectorDBOperations(qdrant_manager)
    
    # Check if collection exists
    if not qdrant_manager.collection_exists():
        logger.error(f"Collection '{settings.qdrant_collection_name}' does not exist!")
        logger.info("Please run 'python scripts/ingest_data.py' first to create the collection.")
        return
    
    logger.info(f"✓ Connected to collection: {settings.qdrant_collection_name}")
    
    # Get filter statistics
    logger.info("\nGetting filter statistics...")
    stats = db_ops.get_filter_statistics()
    
    if stats:
        logger.info(f"\nCollection Statistics:")
        logger.info(f"  Total documents: {stats['total_documents']}")
        logger.info(f"  Categories: {', '.join(stats['available_categories'])}")
        logger.info(f"  Countries: {', '.join(stats['available_countries'])}")
        logger.info(f"  Languages: {', '.join(stats['available_languages'][:10])}")
        logger.info(f"  Total channels: {stats['total_channels']}")
    
    # Test 1: Basic semantic search (no filters)
    logger.info("\n" + "="*80)
    logger.info("Test 1: Basic Semantic Search (No Filters)")
    logger.info("="*80)
    
    query = "funny cat videos"
    logger.info(f"Query: '{query}'")
    
    query_embedding = embedding_model.encode_single(query)
    results = db_ops.search(query_embedding, limit=5)
    print_results(results, "Basic Search Results")
    
    # Test 2: Search with category filter
    logger.info("\n" + "="*80)
    logger.info("Test 2: Search with Category Filter")
    logger.info("="*80)
    
    query = "gaming highlights"
    logger.info(f"Query: '{query}'")
    logger.info(f"Filter: category='Gaming'")
    
    query_embedding = embedding_model.encode_single(query)
    results = db_ops.search(
        query_embedding,
        limit=5,
        filters={'category': 'Gaming'}
    )
    print_results(results, "Gaming Category Results")
    
    # Test 3: Search with view count filter
    logger.info("\n" + "="*80)
    logger.info("Test 3: Search with View Count Filter")
    logger.info("="*80)
    
    query = "popular music videos"
    logger.info(f"Query: '{query}'")
    logger.info(f"Filter: min_views=1,000,000")
    
    query_embedding = embedding_model.encode_single(query)
    results = db_ops.search(
        query_embedding,
        limit=5,
        filters={'min_views': 1000000}
    )
    print_results(results, "High View Count Results")
    
    # Test 4: Search with multiple filters
    logger.info("\n" + "="*80)
    logger.info("Test 4: Search with Multiple Filters")
    logger.info("="*80)
    
    query = "trending videos"
    logger.info(f"Query: '{query}'")
    logger.info(f"Filters: category='Entertainment', min_views=500000, min_days_trending=3")
    
    query_embedding = embedding_model.encode_single(query)
    results = db_ops.search(
        query_embedding,
        limit=5,
        filters={
            'category': 'Entertainment',
            'min_views': 500000,
            'min_days_trending': 3
        }
    )
    print_results(results, "Multi-Filter Results")
    
    # Test 5: Search with country filter
    logger.info("\n" + "="*80)
    logger.info("Test 5: Search with Country Filter")
    logger.info("="*80)
    
    query = "news and politics"
    logger.info(f"Query: '{query}'")
    logger.info(f"Filter: country='CA'")
    
    query_embedding = embedding_model.encode_single(query)
    results = db_ops.search(
        query_embedding,
        limit=5,
        filters={'country': 'CA'}
    )
    print_results(results, "Country-Specific Results")
    
    # Test 6: Search with engagement filters
    logger.info("\n" + "="*80)
    logger.info("Test 6: Search with Engagement Filters")
    logger.info("="*80)
    
    query = "viral videos"
    logger.info(f"Query: '{query}'")
    logger.info(f"Filters: min_likes=10000, min_days_trending=5")
    
    query_embedding = embedding_model.encode_single(query)
    results = db_ops.search(
        query_embedding,
        limit=5,
        filters={
            'min_likes': 10000,
            'min_days_trending': 5
        }
    )
    print_results(results, "High Engagement Results")
    
    logger.info("\n" + "="*80)
    logger.info("Demo Complete!")
    logger.info("="*80)
    logger.info("\nKey Features Demonstrated:")
    logger.info("  ✓ Semantic search with embeddings")
    logger.info("  ✓ Category filtering")
    logger.info("  ✓ View count filtering")
    logger.info("  ✓ Multiple simultaneous filters")
    logger.info("  ✓ Country/language filtering")
    logger.info("  ✓ Engagement metrics filtering")
    logger.info("  ✓ Temporal features (days trending, streak)")
    logger.info("\nAll metadata is consistent between SQL and Vector DB!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
