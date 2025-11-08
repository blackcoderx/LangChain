"""
PHASE 2C: RETRIEVAL TESTING
===========================
Test the quality of your indexed vector store.
Query it with different types of questions and see what gets retrieved.

This teaches you:
- How semantic search actually works
- What types of queries work well
- What types of queries struggle
- Why chunking and embedding choices matter
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment
load_dotenv()


# ============================================================================
# LOAD VECTOR STORE
# ============================================================================

def load_vector_store():
    """
    Load the vector store we created in Phase 2A.
    """
    print("\n" + "="*70)
    print("LOADING VECTOR STORE")
    print("="*70)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )
    
    path = "./vector_store"
    
    if not os.path.exists(path):
        print(f"\n✗ Vector store not found at {path}/")
        print("Did you run phase_2a_index_documents.py yet?")
        return None
    
    try:
        vector_store = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"\n✓ Vector store loaded successfully")
        print(f"  Location: {path}/")
        return vector_store
    except Exception as e:
        print(f"\n✗ Error loading vector store: {e}")
        return None


# ============================================================================
# RETRIEVE AND DISPLAY
# ============================================================================

def retrieve_and_display(vector_store, query, k=3):
    """
    Query the vector store and display results.
    """
    print(f"\n{'='*70}")
    print(f"Query: \"{query}\"")
    print(f"{'='*70}")
    
    # Search for similar documents
    results = vector_store.similarity_search(query, k=k)
    
    print(f"\nFound {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Title: {result.metadata['title']}")
        print(f"  Category: {result.metadata['category']}")
        print(f"  Section: {result.metadata['section']}")
        print(f"  Content: {result.page_content[:120]}...")
        print()
    
    return results


# ============================================================================
# TEST DIFFERENT QUERY TYPES
# ============================================================================

def run_query_type_tests(vector_store):
    """
    Test different types of queries to understand what works well.
    """
    print("\n" + "="*70)
    print("TESTING DIFFERENT QUERY TYPES")
    print("="*70)
    
    # Type 1: Direct concept questions
    print("\n" + "-"*70)
    print("TYPE 1: DIRECT CONCEPT QUESTIONS")
    print("-"*70)
    print("These ask directly about concepts in your documents.")
    print("Semantic search usually works VERY WELL for these.\n")
    
    queries_type1 = [
        "What is RAG?",
        "Explain embeddings",
        "How do chains work?",
    ]
    
    for query in queries_type1:
        retrieve_and_display(vector_store, query, k=2)
    
    # Type 2: How-to questions
    print("\n" + "-"*70)
    print("TYPE 2: HOW-TO QUESTIONS")
    print("-"*70)
    print("These ask for practical guidance.")
    print("Semantic search usually works WELL for these.\n")
    
    queries_type2 = [
        "How do I build a RAG application?",
        "How do I use embeddings?",
        "How does semantic search work?",
    ]
    
    for query in queries_type2:
        retrieve_and_display(vector_store, query, k=2)
    
    # Type 3: Vague questions
    print("\n" + "-"*70)
    print("TYPE 3: VAGUE QUESTIONS")
    print("-"*70)
    print("These are broad and non-specific.")
    print("Semantic search may return mixed results.\n")
    
    queries_type3 = [
        "Tell me about LangChain",
        "What should I know?",
        "Explain everything",
    ]
    
    for query in queries_type3:
        retrieve_and_display(vector_store, query, k=2)
    
    # Type 4: Off-topic questions
    print("\n" + "-"*70)
    print("TYPE 4: OFF-TOPIC QUESTIONS")
    print("-"*70)
    print("These are about things NOT in your documents.")
    print("Semantic search will still return SOMETHING (best match available).")
    print("This shows the limitation: it can't say 'I don't know'.\n")
    
    queries_type4 = [
        "How do I cook pasta?",
        "What's the weather today?",
        "Tell me about space exploration",
    ]
    
    for query in queries_type4:
        retrieve_and_display(vector_store, query, k=2)


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_mode(vector_store):
    """
    Let the user ask custom queries.
    """
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("\nEnter your own queries to test retrieval.")
    print("Type 'quit' to exit.\n")
    
    while True:
        query = input("Query> ").strip()
        
        if query.lower() == "quit":
            print("Exiting interactive mode...")
            break
        
        if not query:
            print("Please enter a query.")
            continue
        
        retrieve_and_display(vector_store, query, k=3)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("PHASE 2C: RETRIEVAL TESTING")
    print("="*70)
    print("\nTest your indexed vector store.")
    print("See what documents are retrieved for different types of queries.")
    
    # Load vector store
    vector_store = load_vector_store()
    if vector_store is None:
        return
    
    # Run tests
    print("\n" + "="*70)
    print("WHAT YOU'LL LEARN")
    print("="*70)
    print("""
This test shows you:
1. How semantic search works in practice
2. What queries work well (direct concepts)
3. What queries struggle (vague, off-topic)
4. Why this matters for Phase 3 (agents can do better)

Pay attention to:
- Are the retrieved documents relevant?
- Do results make sense semantically?
- What happens with off-topic queries?
- How does query specificity affect results?
    """)
    
    # Ask user what to do
    print("\n" + "="*70)
    print("CHOOSE MODE")
    print("="*70)
    print("1. Run automated tests (shows all query types)")
    print("2. Interactive mode (ask your own questions)")
    print("3. Exit")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        run_query_type_tests(vector_store)
        
        print("\n" + "="*70)
        print("TEST COMPLETE")
        print("="*70)
        print("""
KEY OBSERVATIONS:
- Type 1 (direct concepts): Excellent results
- Type 2 (how-to): Good results
- Type 3 (vague): Mixed results
- Type 4 (off-topic): Returns something, but wrong

This is why Phase 3 introduces agents:
Agents can recognize when a query is off-topic and refuse to answer.
Chains (what we use now) always return something.
        """)
        
    elif choice == "2":
        interactive_mode(vector_store)
        
    elif choice == "3":
        print("Exiting...")
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()