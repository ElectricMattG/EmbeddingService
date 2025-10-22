"""
Test script for EmbeddingEngine to check model loading and basic functionality.
"""
import sys
import logging
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from components.embedding_engine import EmbeddingEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_engine():
    """Test the embedding engine."""
    print("=" * 60)
    print("Testing EmbeddingEngine")
    print("=" * 60)
    
    try:
        print("\n1. Initializing engine with 'all-mini' model...")
        engine = EmbeddingEngine(model_name='all-mini')
        
        print("\n2. Getting model info...")
        model_info = await engine.get_model_info()
        print(f"Model Info:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        print("\n3. Testing single text embedding...")
        test_text = "This is a test sentence."
        embedding = await engine.embed(test_text)
        print(f"Generated embedding shape: {embedding.shape}")
        print(f"First 5 values: {embedding[0][:5]}")
        
        print("\n4. Testing batch embeddings...")
        test_texts = [
            "The cat sat on the mat.",
            "A dog ran in the park.",
            "Python is a programming language."
        ]
        embeddings = await engine.embed(test_texts)
        print(f"Generated {len(test_texts)} embeddings")
        print(f"Embeddings shape: {embeddings.shape}")
        
        print("\n5. Testing similarity...")
        query = "A feline on a rug"
        similarities = await engine.similarity([query], test_texts)
        print(f"Query: '{query}'")
        print("Similarities:")
        for i, text in enumerate(test_texts):
            print(f"  {similarities[0][i]:.4f}: {text}")
        
        print("\n6. Finding most similar...")
        similar = await engine.find_most_similar(query, test_texts, top_k=2)
        print("Top 2 most similar:")
        for idx, text, score in similar:
            print(f"  [{idx}] {score:.4f}: {text}")
        
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Error during testing: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_engine())
