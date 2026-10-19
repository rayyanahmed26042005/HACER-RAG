import json

with open('benchmark/rag_index_device.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total chunks in device index: {len(data['entries'])}")
for i, entry in enumerate(data['entries']):
    txt = entry['chunk']['text']
    print(f"--- Chunk {i} ({len(txt)} chars) ---")
    print(txt[:150].replace('\n', ' ') + "...")
