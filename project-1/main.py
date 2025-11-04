import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from synthetic_data import get_synthetic_documents
from components import chunk, embedding



# Load environment variables
load_dotenv()


# ============================================================================
# SECTION 1: LOAD DOCUMENTS
# ============================================================================

def load_documents():
    """
    Convert raw synthetic data into LangChain Document objects.
    
    Each Document has:
    - page_content: The actual text to search
    - metadata: Information about the text (title, category, section)
    """
    print("\n" + "="*70)
    print("SECTION 1: LOADING DOCUMENTS")
    print("="*70)
    
    # Get raw documents from synthetic_data.py
    raw_documents = get_synthetic_documents()
    
    print(f"\n✓ Retrieved {len(raw_documents)} raw documents")
    
    # Convert to LangChain Document objects
    documents = []
    for doc_dict in raw_documents:
        doc = Document(
            page_content=doc_dict["content"].strip(),
            metadata={
                "title": doc_dict["title"],
                "category": doc_dict["category"],
                "section": doc_dict["section"]
            }
        )
        documents.append(doc)
    
    print(f"✓ Converted to {len(documents)} LangChain Document objects")
    
    # Show breakdown by category
    print(f"\nDocuments by category:")
    categories = {}
    for doc in documents:
        cat = doc.metadata["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat in sorted(categories.keys()):
        print(f"  - {cat}: {categories[cat]}")
    
    # Show total size
    total_chars = sum(len(doc.page_content) for doc in documents)
    avg_chars = total_chars // len(documents)
    print(f"\nTotal characters: {total_chars:,}")
    print(f"Average per document: {avg_chars:,} characters")
    
    # Show a sample
    print(f"\nSample document:")
    sample = documents[0]
    print(f"  Title: {sample.metadata['title']}")
    print(f"  Content: {sample.page_content[:100]}...")
    
    return documents


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("PHASE 2A: INDEXING PIPELINE - ALL SECTIONS")
    print("="*70)
    
    documents = load_documents()
    chunks = chunk.chunk_documents(documents)
    vector_store = embedding.create_vector_store(chunks)
    embedding.save_vector_store(vector_store)
    embedding.verify_indexing(vector_store)
    
    print("\n" + "="*70)
    print("✓ INDEXING COMPLETE")
    print("="*70)
    print(f"\nSummary:")
    print(f"  - Documents loaded: {len(documents)}")
    print(f"  - Chunks created: {len(chunks)}")
    print(f"  - Vector store saved: ./vector_store/")
    print(f"\nReady for retrieval and generation!")




if __name__ == "__main__":
    main()