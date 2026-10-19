import json
import math

with open('benchmark/rag_index_device.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Index loaded: {len(data['entries'])} chunks")
for i, e in enumerate(data['entries']):
    vec = [float(value) for value in e['vector']]
    norm = math.sqrt(sum(value * value for value in vec))
    print(f"Chunk {i}: len={len(e['chunk']['text'])} chars, vector dim={len(vec)}, norm={norm:.4f}")
