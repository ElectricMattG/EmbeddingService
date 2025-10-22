#!/usr/bin/env python3
"""
Test script to reproduce the large batch 500 error.
"""
import json

# Create a large batch of 500 texts to test
texts = [f"This is test text number {i} for testing large batch processing with segmentation enabled." for i in range(500)]

payload = {
    "texts": texts,
    "use_segmentation": True
}

# Write to file for curl testing
with open("/tmp/large_batch.json", "w") as f:
    json.dump(payload, f)

print(f"Created payload with {len(texts)} texts")
print(f"JSON size: {len(json.dumps(payload))} bytes ({len(json.dumps(payload))/1024/1024:.2f} MB)")