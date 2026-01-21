#!/usr/bin/env python3
"""
Test script to download and verify LegalBERT model works correctly.
This will download the model and run a simple embedding test.
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def test_legalbert():
    """Test LegalBERT download and basic functionality."""

    print("=" * 80)
    print("Testing LegalBERT Model")
    print("=" * 80)

    model_name = "nlpaueb/legal-bert-base-uncased"

    print(f"\n1. Downloading/Loading model: {model_name}")
    print("   (This may take a few minutes on first run...)")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"   ✓ Tokenizer loaded successfully")

        # Load model
        model = AutoModel.from_pretrained(model_name)
        print(f"   ✓ Model loaded successfully")

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"   ✓ Model moved to device: {device}")

        # Get model info
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ Model parameters: {num_params:,}")

    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return False

    print("\n2. Testing embedding generation")

    # Test texts (legal domain)
    test_texts = [
        "The defendant pleaded guilty to the charges.",
        "The contract was terminated due to breach of clause 5.2.",
        "The court ruled in favor of the plaintiff.",
    ]

    try:
        for i, text in enumerate(test_texts, 1):
            print(f"\n   Test {i}: {text[:50]}...")

            # Tokenize
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            # Generate embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                # Use CLS token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            print(f"   ✓ Embedding shape: {embedding.shape}")
            print(f"   ✓ Embedding norm: {np.linalg.norm(embedding):.4f}")
            print(f"   ✓ Sample values: [{embedding[0, 0]:.4f}, {embedding[0, 1]:.4f}, {embedding[0, 2]:.4f}, ...]")

    except Exception as e:
        print(f"   ✗ Error generating embeddings: {e}")
        return False

    print("\n3. Testing batch processing")

    try:
        # Batch all texts
        inputs = tokenizer(
            test_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        print(f"   ✓ Batch embeddings shape: {embeddings.shape}")
        print(f"   ✓ Expected: ({len(test_texts)}, 768)")

        # Test similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)

        print(f"\n   Cosine similarity matrix:")
        for i in range(len(test_texts)):
            row = " ".join([f"{sim:6.3f}" for sim in similarities[i]])
            print(f"   [{row}]")

    except Exception as e:
        print(f"   ✗ Error in batch processing: {e}")
        return False

    print("\n" + "=" * 80)
    print("✓ All tests passed! LegalBERT is working correctly.")
    print("=" * 80)

    return True


def test_with_embedding_engine():
    """Test using the EmbeddingEngine class."""

    print("\n" + "=" * 80)
    print("Testing LegalBERT with EmbeddingEngine")
    print("=" * 80)

    try:
        import sys
        sys.path.insert(0, '/home/electric/workspace/EmbeddingService/src')

        from components.embedding_engine import EmbeddingEngine

        print("\n1. Initializing EmbeddingEngine with legal-bert")
        engine = EmbeddingEngine(model_name="legal-bert")

        print(f"   ✓ Engine initialized")
        print(f"   ✓ Model: {engine.model_name}")
        print(f"   ✓ Device: {engine.device}")

        print("\n2. Generating embeddings")
        test_texts = [
            "The plaintiff filed a motion to dismiss.",
            "The statute of limitations has expired.",
        ]

        embeddings = engine.encode(test_texts)

        print(f"   ✓ Generated {len(embeddings)} embeddings")
        print(f"   ✓ Embedding shape: {embeddings.shape}")
        print(f"   ✓ Embedding dtype: {embeddings.dtype}")

        # Test similarity
        similarity = np.dot(embeddings[0], embeddings[1])
        print(f"\n3. Similarity between texts: {similarity:.4f}")

        print("\n" + "=" * 80)
        print("✓ EmbeddingEngine test passed!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n✗ Error testing with EmbeddingEngine: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run basic test
    success = test_legalbert()

    if success:
        # Run EmbeddingEngine test
        test_with_embedding_engine()
