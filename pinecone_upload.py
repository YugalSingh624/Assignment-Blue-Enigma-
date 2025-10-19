# pinecone_upload_GEMINI.py
# Upload embeddings using FREE Hugging Face models (no OpenAI!)

import json
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

import config

print("="*70)
print("PINECONE UPLOAD - Using FREE Hugging Face Embeddings")
print("="*70)

# -----------------------------
# Config
# -----------------------------
DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32
INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = 384  # sentence-transformers dimension

# -----------------------------
# Load FREE embedding model
# -----------------------------
print("\n📦 Loading FREE embedding model...")
print("   Model: sentence-transformers/all-MiniLM-L6-v2")
print("   First run will download ~150MB, then cached locally\n")

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✅ Model loaded! Running locally (no API calls needed)\n")

# -----------------------------
# Initialize Pinecone
# -----------------------------
pc = Pinecone(
    api_key=config.PINECONE_API_KEY
)


# Create or connect to index
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    print(f"📌 Creating Pinecone index: {INDEX_NAME}")
    print(f"   Dimension: {VECTOR_DIM}")
    print(f"   Metric: cosine")
    print(f"   Region: us-east1-gcp (FREE tier)\n")
    
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("⏳ Waiting for index to be ready...")
    time.sleep(15)
    print("✅ Index ready!\n")
else:
    print(f"✅ Index '{INDEX_NAME}' already exists\n")

index = pc.Index(INDEX_NAME)

# -----------------------------
# Load data
# -----------------------------
print(f"📂 Loading data from {DATA_FILE}...")
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} nodes\n")

# -----------------------------
# Generate embeddings and upload
# -----------------------------
vectors_to_upsert = []
successful_uploads = 0

print("🔄 Generating embeddings and uploading to Pinecone...")
print("   This runs locally - no API calls!\n")

for item in tqdm(data, desc="Processing nodes"):
    try:
        node_id = item["id"]
        
        # Get text to embed
        text_to_embed = item.get("semantic_text", "") or item.get("description", "")
        
        if not text_to_embed:
            print(f"   ⚠️  Skipping {node_id} - no text found")
            continue
        
        # Generate embedding locally (FREE!)
        embedding = model.encode(text_to_embed, convert_to_tensor=False)
        embedding = embedding.tolist()
        
        # Prepare metadata
        meta = {
            "id": node_id,
            "name": item.get("name", "Unknown"),
            "type": item.get("type", "Unknown"),
            "description": item.get("description", "")[:500],
        }
        
        # Add optional fields
        if "city" in item:
            meta["city"] = item["city"]
        if "region" in item:
            meta["region"] = item["region"]
        if "tags" in item and isinstance(item["tags"], list):
            meta["tags"] = ",".join(item["tags"][:5])
        
        # Add to batch
        vectors_to_upsert.append({
            "id": node_id,
            "values": embedding,
            "metadata": meta
        })
        
        # Upload in batches
        if len(vectors_to_upsert) >= BATCH_SIZE:
            try:
                index.upsert(vectors=vectors_to_upsert)
                successful_uploads += len(vectors_to_upsert)
                vectors_to_upsert = []
            except Exception as e:
                print(f"\n   ❌ Batch upload error: {e}")
                vectors_to_upsert = []
    
    except Exception as e:
        print(f"\n   ❌ Error processing {node_id}: {e}")
        continue

# Upload remaining vectors
if vectors_to_upsert:
    try:
        index.upsert(vectors=vectors_to_upsert)
        successful_uploads += len(vectors_to_upsert)
    except Exception as e:
        print(f"\n   ❌ Final batch upload error: {e}")

print("\n" + "="*70)
print("✅ UPLOAD COMPLETE!")
print("="*70)
print(f"📊 Statistics:")
print(f"   Total nodes processed: {len(data)}")
print(f"   Successfully uploaded: {successful_uploads}")
print(f"   Vector dimension: {VECTOR_DIM}")
print(f"   💰 Total cost: $0.00 (100% FREE!)")
print("="*70)

# Verify upload
try:
    stats = index.describe_index_stats()
    print(f"\n✅ Verification:")
    print(f"   Vectors in index: {stats['total_vector_count']}")
    print(f"   Index ready: ✅")
except Exception as e:
    print(f"\n⚠️  Could not verify: {e}")

print("\n🎉 All done! Ready to run hybrid_chat_GEMINI.py")