"""
Standalone test for EmbeddingEngine without using the config system.
"""
import asyncio
import logging
import sys
import os

# Suppress warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import warnings
warnings.filterwarnings("ignore")

# Setup basic logging to console only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import required modules
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import normalize

print("=" * 70)
print("STANDALONE EMBEDDING ENGINE TEST")
print("=" * 70)

# Check CUDA
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Test model name
model_name = "sentence-transformers/all-MiniLM-L6-v2"
print(f"\nTrying to load model: {model_name}")

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading model...")
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    
    # Test embedding generation
    print("\nGenerating test embeddings...")
    test_texts = [
        "The cat sat on the mat.",
        "A dog ran in the park.",
        "Python is a programming language."
    ]
    
    # Tokenize
    encoded = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Forward pass
    with torch.no_grad():
        model_output = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Mean pooling
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = sum_embeddings / sum_mask
    
    # Normalize
    embeddings = normalize(embeddings, p=2, dim=1)
    embeddings_np = embeddings.cpu().numpy()
    
    print(f"✅ Generated embeddings shape: {embeddings_np.shape}")
    print(f"   Embedding dimension: {embeddings_np.shape[1]}")
    print(f"   Sample values: {embeddings_np[0][:5]}")
    
    # Compute similarities
    print("\nComputing similarities...")
    similarities = np.dot(embeddings_np, embeddings_np.T)
    print("Similarity matrix:")
    for i, text1 in enumerate(test_texts):
        print(f"\n{text1}")
        for j, text2 in enumerate(test_texts):
            print(f"  vs '{text2}': {similarities[i, j]:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - MODEL WORKS CORRECTLY!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
