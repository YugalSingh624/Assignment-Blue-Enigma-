# hybrid_chat_GEMINI.py
# Complete hybrid AI system using Google Gemini + FREE embeddings
# Author: Yugal | Blue Enigma Labs Challenge
# 100% FREE solution!

import json
import asyncio
import hashlib
import logging
from typing import List, Dict, Tuple
from collections import defaultdict
from datetime import datetime

from sentence_transformers import SentenceTransformer
from pinecone.grpc import PineconeGRPC as Pinecone

from neo4j import GraphDatabase
# import google.generativeai as genai
import config

from google import genai
from google.genai import types


# -----------------------------
# Setup Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("="*70)
print("🌏 HYBRID TRAVEL ASSISTANT - Google Gemini Edition")
print("Using: Gemini API + Hugging Face Embeddings + Neo4j + Pinecone")
print("="*70 + "\n")

# -----------------------------
# Config
# -----------------------------
INDEX_NAME = config.PINECONE_INDEX_NAME
TOP_K = 10
RRF_K = 60

# -----------------------------
# Initialize FREE models
# -----------------------------
print("📦 Initializing components...\n")

# 1. Embedding model (FREE, local)
print("   1/4 Loading embedding model...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
logger.info("   ✅ Embedding model ready (local)")

# 2. Google Gemini (FREE API)
print("   2/4 Configuring Google Gemini...")
# genai.configure(api_key=config.GEMINI_API_KEY)
client = genai.Client(api_key=config.GEMINI_API_KEY)
logger.info("   ✅ Gemini API configured")

# 3. Pinecone (FREE tier)
print("   3/4 Connecting to Pinecone...")
pc = Pinecone(api_key=config.PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    logger.error(f"   ❌ Index '{INDEX_NAME}' not found!")
    print(f"\n⚠️  Please run 'python pinecone_upload_GEMINI.py' first!\n")
    exit(1)
index = pc.Index(INDEX_NAME)
logger.info("   ✅ Pinecone connected")

# 4. Neo4j (FREE Aura)
print("   4/4 Connecting to Neo4j...")
try:
    driver = GraphDatabase.driver(
        config.NEO4J_URI, 
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    # Test connection
    with driver.session() as session:
        session.run("RETURN 1")
    logger.info("   ✅ Neo4j connected")
except Exception as e:
    logger.error(f"   ❌ Neo4j connection failed: {e}")
    print(f"\n⚠️  Please check Neo4j credentials in config.py\n")
    exit(1)

print("\n✅ All systems ready!\n")

# -----------------------------
# INNOVATION: Embedding Cache
# -----------------------------
class EmbeddingCache:
    """In-memory cache to reduce embedding computation."""
    
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _hash_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()
    
    def get(self, text: str):
        key = self._hash_key(text)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, text: str, embedding):
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        key = self._hash_key(text)
        self.cache[key] = embedding
    
    def hit_rate(self):
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0
    
    def stats(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate():.2f}%",
            "size": len(self.cache)
        }

embedding_cache = EmbeddingCache()

# -----------------------------
# Helper functions
# -----------------------------
def embed_text(text: str) -> List[float]:
    """Generate embedding with caching."""
    cached = embedding_cache.get(text)
    if cached is not None:
        return cached
    
    try:
        embedding = embedding_model.encode(text, convert_to_tensor=False)
        embedding = embedding.tolist()
        embedding_cache.set(text, embedding)
        return embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise

async def async_pinecone_query(query_text: str, top_k=TOP_K):
    """Async Pinecone query."""
    try:
        vec = embed_text(query_text)
        loop = asyncio.get_event_loop()
        
        res = await loop.run_in_executor(
            None,
            lambda: index.query(
                vector=vec,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
        )
        
        logger.info(f"Pinecone: {len(res['matches'])} results")
        return res["matches"]
    except Exception as e:
        logger.error(f"Pinecone error: {e}")
        return []

async def async_fetch_graph_context(node_ids: List[str]):
    """Async Neo4j query with UNWIND optimization."""
    try:
        cypher_query = """
        UNWIND $node_ids AS nid
        MATCH (n:Entity {id: nid})-[r]-(m:Entity)
        RETURN nid AS source, type(r) AS rel,
               m.id AS id, m.name AS name, m.type AS type,
               m.description AS description
        LIMIT 50
        """
        
        loop = asyncio.get_event_loop()
        
        def run_neo4j():
            with driver.session() as session:
                result = session.run(cypher_query, node_ids=node_ids)
                return list(result)
        
        records = await loop.run_in_executor(None, run_neo4j)
        
        facts = []
        for r in records:
            facts.append({
                "source": r["source"],
                "rel": r["rel"],
                "target_id": r["id"],
                "target_name": r["name"],
                "target_type": r.get("type", "Unknown"),
                "target_desc": (r["description"] or "")[:300]
            })
        
        logger.info(f"Neo4j: {len(facts)} relationships")
        return facts
    
    except Exception as e:
        logger.error(f"Neo4j error: {e}")
        return []

# -----------------------------
# INNOVATION: Reciprocal Rank Fusion
# -----------------------------
def reciprocal_rank_fusion(
    vector_results: List[Dict],
    graph_facts: List[Dict],
    k: int = RRF_K
) -> List[Tuple[str, float]]:
    """Merge results using RRF algorithm."""
    scores = defaultdict(float)
    
    # Score vector results
    for rank, match in enumerate(vector_results):
        doc_id = match["id"]
        scores[doc_id] += 1 / (k + rank)
    
    # Score unique graph targets
    unique_targets = {}
    for fact in graph_facts:
        target_id = fact["target_id"]
        if target_id not in unique_targets:
            unique_targets[target_id] = fact
    
    for rank, target_id in enumerate(unique_targets.keys()):
        scores[target_id] += 1 / (k + rank)
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    logger.info(f"RRF: {len(ranked)} combined results")
    return ranked

# -----------------------------
# INNOVATION: Enhanced Prompts with Gemini
# -----------------------------
def build_gemini_prompt(user_query, matches, graph_facts, fused_results):
    """Build enhanced prompt for Gemini."""
    
    # Format semantic matches
    vec_context = []
    for m in matches[:8]:
        meta = m["metadata"]
        vec_context.append(
            f"• {meta.get('name', 'Unknown')} ({meta.get('type', 'N/A')})\n"
            f"  ID: {m['id']}\n"
            f"  Description: {meta.get('description', '')[:200]}\n"
            f"  Relevance: {m.get('score', 0):.3f}"
        )
    
    # Format graph relationships
    graph_context = []
    for f in graph_facts[:20]:
        graph_context.append(
            f"• {f['source']} --[{f['rel']}]--> {f['target_name']} ({f['target_type']})"
        )
    
    # Format top fused results
    fusion_context = []
    for node_id, score in fused_results[:10]:
        fusion_context.append(f"• {node_id}: {score:.4f}")
    
    prompt = f"""You are an expert travel assistant for Vietnam. Analyze the following information and provide helpful recommendations.

USER QUERY: {user_query}

SEMANTIC MATCHES (from vector search):
{chr(10).join(vec_context) if vec_context else "No matches found"}

GRAPH RELATIONSHIPS (from knowledge graph):
{chr(10).join(graph_context) if graph_context else "No relationships found"}

TOP COMBINED RESULTS (RRF scores):
{chr(10).join(fusion_context) if fusion_context else "No results"}

INSTRUCTIONS:
1. Analyze the user's travel preferences and constraints
2. Review both semantic matches and graph relationships
3. Synthesize the information to create practical recommendations
4. Provide relevant suggestions with node IDs for reference
5. Consider practical logistics (connections, travel time, etc.)
6. Give all the relevant information in your response

Generate a helpful response now:"""
    
    return prompt

async def call_gemini_async(prompt: str):
    try:
        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=genai.types.GenerateContentConfig(
                temperature=0.3,
                # max_output_tokens=800,
            )
        )
        if hasattr(response, "text") and response.text:
            return response.text

        # Fallback for edge cases
        parts = getattr(response.candidates, "parts", [])
        texts = [p.text for p in parts if hasattr(p, "text")]
        if texts:
            return "".join(texts)

        return "[No text returned by the model]"
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return "I encountered an error generating the response."



# -----------------------------
# INNOVATION: Async Parallel Retrieval
# -----------------------------
async def hybrid_retrieve(query: str):
    """
    Complete hybrid retrieval with:
    - Async parallel operations
    - Embedding caching
    - RRF fusion
    - Gemini generation
    """
    logger.info(f"Query: {query}")
    start_time = datetime.now()
    
    # Parallel retrieval
    matches = await async_pinecone_query(query, top_k=TOP_K)
    
    if not matches:
        return {
            "answer": "I couldn't find relevant information. Try rephrasing your query.",
            "matches": [],
            "facts": [],
            "fused": [],
            "elapsed": 0
        }
    
    match_ids = [m["id"] for m in matches]
    graph_facts = await async_fetch_graph_context(match_ids)
    
    # Apply RRF
    fused_results = reciprocal_rank_fusion(matches, graph_facts)
    
    # Build prompt and generate with Gemini
    prompt = build_gemini_prompt(query, matches, graph_facts, fused_results)
    answer = await call_gemini_async(prompt)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Total time: {elapsed:.2f}s")
    
    return {
        "answer": answer,
        "matches": matches,
        "facts": graph_facts,
        "fused": fused_results[:10],
        "elapsed": elapsed
    }

# -----------------------------
# Interactive Chat
# -----------------------------
async def interactive_chat():
    print("="*70)
    print("💬 CHAT READY - Ask me anything about Vietnam travel!")
    print("="*70)
    print("Commands:")
    print("  • Type your question to get recommendations")
    print("  • 'stats' - View cache statistics")
    print("  • 'exit' - Quit")
    print("="*70 + "\n")
    
    while True:
        try:
            query = input("🌏 You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ("exit", "quit", "q"):
                print("\n👋 Goodbye! Safe travels!")
                print(f"Final cache stats: {embedding_cache.stats()}\n")
                break
            
            if query.lower() == "stats":
                stats = embedding_cache.stats()
                print(f"\n📊 Cache Statistics:")
                print(f"   Hits: {stats['hits']}")
                print(f"   Misses: {stats['misses']}")
                print(f"   Hit Rate: {stats['hit_rate']}")
                print(f"   Cache Size: {stats['size']}\n")
                continue
            
            print("\n⏳ Processing (async retrieval + Gemini)...\n")
            
            result = await hybrid_retrieve(query)
            
            print("="*70)
            print("🤖 Assistant:")
            print("="*70)
            print(result["answer"])
            print("\n" + "="*70)
            print(f"⚡ Time: {result['elapsed']:.2f}s | Cache: {embedding_cache.stats()['hit_rate']}")
            print("="*70 + "\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\n❌ An error occurred: {e}\n")
            continue

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    try:
        asyncio.run(interactive_chat())
    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        driver.close()
        logger.info("Connections closed")