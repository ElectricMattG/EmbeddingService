#!/usr/bin/env python3
"""
Test script to reproduce the segmentation issue.
"""
import asyncio
import traceback

from src.components.embedding_engine import EmbeddingEngine


async def test_segmentation():
    """Test embedding with segmentation enabled."""
    try:
        print("Initializing EmbeddingEngine...")
        engine = EmbeddingEngine(model_name="all-mini")

        print("Testing with segmentation=False...")
        result1 = await engine.embed(
            ["This is a test text to check functionality."],
            use_segmentation=False
        )
        print(f"Success without segmentation: shape={result1.shape}")

        print("Testing with segmentation=True...")
        result2 = await engine.embed(
            ["This is a test text to check segmentation functionality."],
            use_segmentation=True
        )
        print(f"Success with segmentation: shape={result2.shape}")

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_segmentation())
