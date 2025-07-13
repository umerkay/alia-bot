from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools import Tool
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import pickle
from langchain_core.tools import tool
import os
from langchain_core.runnables import RunnableConfig
from pathlib import Path
from app.config import settings


embeddings = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL)
vectorstore = Chroma(
    collection_name=settings.CHROMA_COLLECTION_NAME, 
    persist_directory=str(settings.chroma_db_path), 
    embedding_function=embeddings
)

# Global retriever variable
retriever = None

# def load_bm25_index():
#     """Load BM25 index from pickle file."""
#     bm25_index_path = settings.bm25_index_path
    
#     if not bm25_index_path.exists():
#         raise FileNotFoundError(f"BM25 index file not found: {bm25_index_path}")
    
#     with open(bm25_index_path, "rb") as f:
#         data = pickle.load(f)
    
#     # Recreate BM25 retriever from saved documents
#     bm25_retriever = BM25Retriever.from_documents(data["docs"])
    
#     # Restore configuration
#     if "retriever_state" in data:
#         bm25_retriever.k = data["retriever_state"].get("k", settings.BM25_K)
#     else:
#         bm25_retriever.k = settings.BM25_K
    
#     return bm25_retriever


async def create_vectorstore():
    """Load existing vectorstore and BM25 index."""
    global retriever
    
    # Create vector retriever from existing Chroma DB
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.VECTOR_SEARCH_K},
    )
    
    # Load BM25 retriever from pickle file
    # try:
        # keyword_retriever = load_bm25_index()
        # print(f"Loaded BM25 index with k={keyword_retriever.k}")
        
        # Create ensemble retriever
        # retriever = EnsembleRetriever(
        #     retrievers=[vector_retriever, keyword_retriever],
        #     weights=[settings.ENSEMBLE_VECTOR_WEIGHT, settings.ENSEMBLE_BM25_WEIGHT],
        # )
        
        # For now, use only vector retriever (as in original code)
        # retriever = vector_retriever
        
    # except FileNotFoundError as e:
    #     print(f"Warning: {e}")
    #     print("Using only vector retriever without BM25")
    #     retriever = vector_retriever
    # except Exception as e:
    #     print(f"Error loading BM25 index: {e}")
    #     print("Using only vector retriever without BM25")
    #     retriever = vector_retriever

    retriever = vector_retriever  # For now, only use vector retriever

def vectorstore_retrieval(query: str, patient_id: str, document_type: str) -> str:
        # print(f"Query: {query}")
        """Retrieves relevant documents from the vectorstore based on the query."""

        retriever.search_kwargs["filter"] = retriever.search_kwargs["filter"] = {
            "$and": [
                {"patient_id": {"$eq": patient_id}},
                {"document_type": {"$eq": document_type}}
            ]
        }
        results = retriever.invoke(query)
        print(query)
        print(results)
        if not results:
            return "No relevant documents found."
        contents = results[0].metadata["source"] + ("="*10) + "\n"
        contents += f"is the most relevant. Found total {len(results)} relevant document chunks.\n"
        contents += "\n".join([( ("="*10) + "\n" + doc.metadata["source"] + "\n" + ("="*10) + "\n" +
            doc.page_content) for doc in results])
        
        return contents        

def create_transcript_retrieval_tool():

    @tool
    def rag_retriever(query: str, config: RunnableConfig) -> str:
        """Searches therapy session transcripts for the given query using semantic retrieval (RAG).

        Args:
            query (str): The query to search for.

        Returns:
            str: The retrieved information.
        """
        
        try:
            patient_id = config["configurable"].get("patient_id")
            return vectorstore_retrieval(query, patient_id=patient_id, document_type="session_transcript")
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return f"An error occurred during retrieval of documents"

    return rag_retriever

def create_intake_form_retrieval_tool():

    @tool
    def intake_form_retriever(query: str, config: RunnableConfig) -> str:
        """Searches intake form information for the given query using semantic retrieval (RAG).

        Args:
            query (str): The query to search for.

        Returns:
            str: The retrieved information.
        """
        
        try:
            patient_id = config["configurable"].get("patient_id")
            return vectorstore_retrieval(query, patient_id=patient_id, document_type="intake_form")
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return f"An error occurred during retrieval of intake form documents"

    return intake_form_retriever