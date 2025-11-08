"""
PHASE 2D: RAG CHAIN
==================
Combine retrieval with generation to answer questions.

This is the complete RAG system:
1. Retrieve relevant documents from vector store
2. Pass them as context to the LLM
3. LLM generates an answer grounded in the documents
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval import create_retrieval_chain
# from langchain.chains.retrieval import create_retrieval_chain # type: ignore
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain # type: ignore
from langchain_core.prompts import ChatPromptTemplate

# Load environment
load_dotenv()


# ============================================================================
# LOAD COMPONENTS
# ============================================================================

def load_rag_components():
    """
    Load all components needed for RAG:
    1. Embeddings (same model as used for indexing)
    2. Vector store (our indexed documents)
    3. Retriever (tool to find relevant documents)
    4. LLM (model to generate answers)
    """
    print("\n" + "="*70)
    print("LOADING RAG COMPONENTS")
    print("="*70)
    
    # 1. Load embeddings
    print("\n1. Loading embeddings model...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )
    print("   ✓ Embeddings model loaded")
    
    # 2. Load vector store
    print("\n2. Loading vector store...")
    path = "../vector_store"
    if not os.path.exists(path):
        print(f"   ✗ Vector store not found at {path}/")
        print("   Run phase_2a_index_documents.py first")
        return None, None, None
    
    try:
        vector_store = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"   ✓ Vector store loaded")
    except Exception as e:
        print(f"   ✗ Error loading vector store: {e}")
        return None, None, None
    
    # 3. Create retriever from vector store
    print("\n3. Creating retriever...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print("   ✓ Retriever created (will fetch top 3 documents)")
    
    # 4. Initialize LLM
    print("\n4. Initializing LLM...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.5
    )
    print("   ✓ LLM initialized (Gemini 2.5 Flash)")
    
    return retriever, llm, vector_store


# ============================================================================
# CREATE RAG CHAIN
# ============================================================================

def create_rag_chain(retriever, llm):
    """
    Create a complete RAG chain:
    1. Define the prompt template (how to format context and question)
    2. Create document chain (formats retrieved documents)
    3. Create retrieval chain (combines retrieval + generation)
    """
    print("\n" + "="*70)
    print("CREATING RAG CHAIN")
    print("="*70)
    
    # Define the system prompt
    system_prompt = """You are an expert on LangChain, a framework for building LLM applications.

Answer the user's question based ONLY on the provided context.
If the context doesn't contain enough information to answer, say "I don't have enough information to answer this question."
Always cite which document your answer comes from.

Context from documents:
{context}"""

    # Create prompt template
    print("\n1. Creating prompt template...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    print("   ✓ Prompt template created")
    
    # Create document chain (formats retrieved documents into the prompt)
    print("\n2. Creating document chain...")
    document_chain = create_stuff_documents_chain(llm, prompt)
    print("   ✓ Document chain created")
    
    # Create retrieval chain (combines retriever + document chain)
    print("\n3. Creating retrieval chain...")
    rag_chain = create_retrieval_chain(retriever, document_chain)
    print("   ✓ RAG chain created")
    
    print("\nChain architecture:")
    print("  Question → Retriever → Top 3 documents →")
    print("  Format into prompt → LLM → Answer with citations")
    
    return rag_chain


# ============================================================================
# ANSWER QUESTIONS
# ============================================================================

def answer_question(rag_chain, question):
    """
    Ask a question and get an answer from the RAG chain.
    """
    print(f"\n{'='*70}")
    print(f"Question: {question}")
    print(f"{'='*70}")
    
    # Run the chain
    result = rag_chain.invoke({"input": question})
    
    # Extract answer
    answer = result.get("answer", "No answer generated")
    
    # Extract source documents
    context_docs = result.get("context", [])
    
    # Display answer
    print(f"\nAnswer:")
    print(f"{answer}")
    
    # Display sources
    print(f"\nSources (retrieved documents):")
    print("-" * 70)
    for i, doc in enumerate(context_docs, 1):
        print(f"\n{i}. {doc.metadata['title']} ({doc.metadata['category']})")
        print(f"   {doc.page_content[:100]}...")
    
    print()


# ============================================================================
# DEMO MODE
# ============================================================================

def run_demo(rag_chain):
    """
    Run demonstration with pre-defined questions.
    """
    print("\n" + "="*70)
    print("DEMO: ANSWERING QUESTIONS WITH RAG")
    print("="*70)
    
    demo_questions = [
        "What is RAG and why is it useful?",
        "How do embeddings work?",
        "What's the difference between chains and agents?",
        "Explain the indexing phase of RAG",
    ]
    
    for i, question in enumerate(demo_questions, 1):
        answer_question(rag_chain, question)
        
        if i < len(demo_questions):
            input("Press Enter to continue to next question...")


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def run_interactive(rag_chain):
    """
    Let user ask custom questions.
    """
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("\nAsk questions about LangChain.")
    print("The system will retrieve relevant documents and generate answers.")
    print("Type 'quit' to exit.\n")
    
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() == "quit":
            print("Exiting...")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        answer_question(rag_chain, question)


# ============================================================================
# ANALYSIS MODE
# ============================================================================

def run_analysis(rag_chain):
    """
    Test different types of questions to understand RAG behavior.
    """
    print("\n" + "="*70)
    print("ANALYSIS: HOW RAG HANDLES DIFFERENT QUESTIONS")
    print("="*70)
    
    test_cases = {
        "Direct concept questions": [
            "What is a chain?",
            "Define embeddings",
        ],
        "How-to questions": [
            "How do I build a RAG application?",
            "How do retrievers work?",
        ],
        "Complex questions": [
            "Explain how RAG, agents, and tools work together",
            "What are the steps to build a production RAG system?",
        ],
        "Out-of-domain questions": [
            "How do I cook pasta?",
            "What's the capital of France?",
        ],
    }
    
    for category, questions in test_cases.items():
        print(f"\n" + "-"*70)
        print(f"Category: {category}")
        print("-"*70)
        
        for question in questions:
            answer_question(rag_chain, question)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("PHASE 2D: RAG CHAIN")
    print("="*70)
    print("\nCombine retrieval with generation to answer questions.")
    
    # Load components
    print("\nLoading components...")
    retriever, llm, vector_store = load_rag_components()
    
    if retriever is None:
        return
    
    # Create RAG chain
    rag_chain = create_rag_chain(retriever, llm)
    
    # Ask user what mode
    print("\n" + "="*70)
    print("CHOOSE MODE")
    print("="*70)
    print("1. Demo mode (predefined questions)")
    print("2. Interactive mode (ask your own questions)")
    print("3. Analysis mode (test different question types)")
    print("4. Exit")
    
    choice = input("\nEnter choice (1, 2, 3, or 4): ").strip()
    
    if choice == "1":
        run_demo(rag_chain)
        
        print("\n" + "="*70)
        print("DEMO COMPLETE")
        print("="*70)
        print("""
WHAT YOU JUST SAW:
- Questions were converted to embeddings
- Most similar documents were retrieved
- Retrieved documents were formatted as context
- LLM generated answers using that context
- Answers were grounded in your documents

This is production RAG in action!
        """)
        
    elif choice == "2":
        run_interactive(rag_chain)
        
    elif choice == "3":
        run_analysis(rag_chain)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print("""
KEY INSIGHTS:
- Direct questions: Excellent answers with proper citations
- How-to questions: Good answers using multiple documents
- Complex questions: Good answers combining multiple concepts
- Out-of-domain questions: Still generates answers (using closest matches)

The limitation: RAG can't refuse to answer out-of-domain questions.
Solution: Phase 3 uses agents that can recognize and refuse.
        """)
    
    elif choice == "4":
        print("Exiting...")
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()