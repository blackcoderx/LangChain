# ============================================================================
# SECTION 2: CHUNK DOCUMENTS
# ============================================================================
from langchain_text_splitters import CharacterTextSplitter

def chunk_documents(documents):
    """
    Split documents into smaller chunks.
    
    Parameters:
    - chunk_size=500: Target size ~500 characters per chunk
    - chunk_overlap=50: 50 character overlap between chunks
    - separator="\n\n": Split on paragraph boundaries
    """
    
    print("\n" + "="*70)
    print("SECTION 2: CHUNKING DOCUMENTS")
    print("="*70)
    
    # Initialize the text splitter
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    
    print(f"\nSplitter configuration:")
    print(f"  - Separator: paragraph breaks (\\n\\n)")
    print(f"  - Chunk size: 500 characters")
    print(f"  - Overlap: 50 characters")
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)
    
    print(f"\n✓ Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Calculate statistics
    chunk_sizes = [len(chunk.page_content) for chunk in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes)
    min_size = min(chunk_sizes)
    max_size = max(chunk_sizes)
    
    print(f"\nChunk statistics:")
    print(f"  - Average size: {avg_size:.0f} characters")
    print(f"  - Minimum size: {min_size} characters")
    print(f"  - Maximum size: {max_size} characters")
    print(f"  - Total data: {sum(chunk_sizes):,} characters")
    
    # Show a sample chunk
    print(f"\nSample chunk (from '{chunks[0].metadata['title']}'):")
    print(f"  Category: {chunks[0].metadata['category']}")
    print(f"  Content: {chunks[0].page_content[:150]}...")
    
    return chunks