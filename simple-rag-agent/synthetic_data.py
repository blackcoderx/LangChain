"""
SYNTHETIC DATA FOR LEARNING
============================
These are 10 fictional but realistic documents about LangChain.
We created them to teach RAG indexing without network complexity.

In production, you'd load from real sources (websites, PDFs, databases).
For learning, synthetic data is perfect.
"""

# These are invented documents that mimic real technical documentation
SYNTHETIC_DOCUMENTS = [
    {
        "title": "Introduction to LangChain",
        "content": """
LangChain is a framework for developing applications powered by language models. 
It enables applications that are data-aware and agentic. 
LangChain provides a standard interface for language models, chat models, and embeddings.
The core idea is to standardize how you interact with models so you can seamlessly swap providers.
This prevents vendor lock-in and allows rapid iteration across different model providers.
        """,
        "category": "basics",
        "section": "introduction"
    },
    {
        "title": "Understanding Chains",
        "content": """
A chain in LangChain is a sequence of calls to language models or other tools.
The simplest chain takes a prompt template, formats it with user input, and passes it to an LLM.
Chains can be composed together - the output of one chain becomes the input to another.
This compositional approach allows building complex workflows from simple pieces.
Common chain types include LLMChain, SequentialChain, and Router chains.
        """,
        "category": "concepts",
        "section": "chains"
    },
    {
        "title": "Retrieval Augmented Generation (RAG)",
        "content": """
RAG combines retrieval with generation. When a user asks a question, relevant documents are 
retrieved from a knowledge base and provided as context to the language model.
The language model then generates an answer using both the question and retrieved context.
This approach makes models more accurate because they're grounded in specific documents.
RAG enables models to answer questions about private data without fine-tuning.
The three phases of RAG are: indexing (offline), retrieval (per query), and generation (per query).
        """,
        "category": "concepts",
        "section": "rag"
    },
    {
        "title": "Document Loaders",
        "content": """
Document loaders are LangChain components that read documents from various sources.
LangChain supports loading from PDFs, text files, web pages, databases, and more.
Each loader handles the specifics of its data source, extracting text and metadata.
After loading, documents go through chunking and embedding before indexing.
The DocumentLoader interface is standardized, so loaders are interchangeable.
        """,
        "category": "components",
        "section": "loaders"
    },
    {
        "title": "Text Splitting and Chunking",
        "content": """
Raw documents are often too large to process at once. Chunking breaks documents into smaller pieces.
Different chunking strategies exist: by character count, by tokens, by semantic boundaries, or hierarchically.
Chunking size is a trade-off: larger chunks have more context but waste tokens on irrelevant information.
Smaller chunks are more precise but might split coherent information across multiple chunks.
LangChain provides TextSplitter classes like CharacterTextSplitter and RecursiveCharacterTextSplitter.
        """,
        "category": "components",
        "section": "chunking"
    },
    {
        "title": "Embeddings and Vector Stores",
        "content": """
Embeddings convert text into numerical vectors. Semantically similar text produces similar vectors.
This enables semantic search: convert a query to an embedding and find closest document embeddings.
Vector stores are databases optimized for searching high-dimensional embeddings.
Common vector stores include FAISS (in-memory), Pinecone (cloud), Weaviate, and Qdrant.
LangChain provides a standard VectorStore interface, making vector stores interchangeable.
        """,
        "category": "components",
        "section": "embeddings"
    },
    {
        "title": "Building Simple RAG Applications",
        "content": """
A minimal RAG application has three components: a document loader, a vector store with embeddings, and an LLM.
First, load and chunk your documents. Second, create embeddings and store them in a vector store.
Third, when a query arrives, retrieve relevant documents and pass them with the query to the LLM.
The LLM generates an answer grounded in the retrieved documents.
This pattern works for Q&A applications, documentation assistants, and knowledge base systems.
        """,
        "category": "tutorial",
        "section": "simple-rag"
    },
    {
        "title": "Agents and Tool Use",
        "content": """
Agents are language models that can use tools to accomplish tasks.
Unlike chains which follow a predefined sequence, agents decide what steps to take.
An agent receives a task, decides if it needs external information or tools, and iterates until it has an answer.
Tools can be retrievers (for RAG), calculators, web search, database queries, or any other capability.
Agents enable flexible, multi-step reasoning that chains cannot provide.
        """,
        "category": "concepts",
        "section": "agents"
    },
    {
        "title": "Agentic RAG",
        "content": """
Agentic RAG combines agents with retrieval. Instead of always retrieving before generating,
an agent decides when retrieval is needed and what to search for.
This is more efficient because some questions don't need external information.
For complex questions, an agent can retrieve multiple times and reason across results.
Agentic RAG enables sophisticated question-answering that pure RAG chains cannot handle.
        """,
        "category": "advanced",
        "section": "agentic-rag"
    },
    {
        "title": "Memory and Conversation",
        "content": """
Memory in LangChain enables multi-turn conversations where context from previous interactions persists.
Different memory types exist: BufferMemory (stores all messages), SummaryMemory (compresses history), EntityMemory.
Memory is typically passed to chains or agents so they can reference prior context.
For production systems, memory is stored in databases, not just in memory (which would be lost on restart).
LangChain provides memory abstractions that integrate with various storage backends.
        """,
        "category": "advanced",
        "section": "memory"
    }
]


def get_synthetic_documents():
    """Return all synthetic documents as a list of dictionaries"""
    return SYNTHETIC_DOCUMENTS


def get_document_by_category(category):
    """Filter documents by category - useful for testing specific areas"""
    return [doc for doc in SYNTHETIC_DOCUMENTS if doc["category"] == category]


def get_document_count():
    """Return the total number of synthetic documents"""
    return len(SYNTHETIC_DOCUMENTS)


if __name__ == "__main__":
    # This runs when you execute this file directly
    # It's a quick test to verify the data loads correctly
    
    print(f"Total synthetic documents: {get_document_count()}")
    print("\nDocuments by category:")
    
    categories = set(doc["category"] for doc in SYNTHETIC_DOCUMENTS)
    for cat in sorted(categories):
        count = len(get_document_by_category(cat))
        print(f"  - {cat}: {count} documents")
    
    print("\nSample document:")
    sample = SYNTHETIC_DOCUMENTS[0]
    print(f"  Title: {sample['title']}")
    print(f"  Category: {sample['category']}")
    print(f"  Content preview: {sample['content'][:100]}...")