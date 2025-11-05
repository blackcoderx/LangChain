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
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)
    
    
    return chunks