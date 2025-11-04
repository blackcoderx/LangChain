# ============================================================================
# SECTION 3: CREATE EMBEDDINGS AND VECTOR STORE
# ============================================================================

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):
    """
    Create embeddings for all chunks and store in FAISS vector database.
    
    This calls Google's Gemini API to convert text to vectors.
    Each chunk becomes a 768-dimensional vector.
    """

    
    print("\n" + "="*70)
    print("SECTION 3: CREATING EMBEDDINGS AND VECTOR STORE")
    print("="*70)
    
    # Initialize embeddings model
    print("\n1. Initializing embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )
    print("   ✓ Embedding model initialized")
    print("   - Model: Google Gemini Embedding 001")
    print("   - Dimensions: 768")
    
    # Create vector store from chunks
    print("\n2. Creating embeddings for all chunks...")
    print(f"   Processing {len(chunks)} chunks (this takes 1-2 minutes)...")
    
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        print(f"   ✓ Vector store created successfully")
        print(f"   - Total vectors: {len(chunks)}")
        print(f"   - Storage type: FAISS (in-memory)")
        
    except Exception as e:
        print(f"   ✗ Error creating vector store: {e}")
        raise
    
    return vector_store
  
  
  # ============================================================================
# SECTION 4: SAVE VECTOR STORE TO DISK
# ============================================================================

def save_vector_store(vector_store, path="./vector_store"):
    """
    Save the vector store to disk for future use.
    """
    print("\n" + "="*70)
    print("SECTION 4: SAVING VECTOR STORE TO DISK")
    print("="*70)
    
    vector_store.save_local(path)
    
    print(f"\n✓ Vector store saved to {path}/")
    print(f"\nFiles created:")
    print(f"  - {path}/index.faiss")
    print(f"  - {path}/index.pkl")
    
    
    
# ============================================================================
# SECTION 5: VERIFICATION
# ============================================================================

def verify_indexing(vector_store):
    """
    Test that indexing worked by doing a sample search.
    """
    print("\n" + "="*70)
    print("SECTION 5: VERIFICATION")
    print("="*70)
    
    # Test semantic search
    test_query = "What is RAG?"
    print(f"\nTest query: \"{test_query}\"")
    print("Searching for most similar chunks...")
    
    results = vector_store.similarity_search(test_query, k=3)
    
    print(f"\n✓ Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"    Title: {result.metadata['title']}")
        print(f"    Content: {result.page_content[:]}...")