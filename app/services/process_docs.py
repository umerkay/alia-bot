from fastapi import HTTPException, UploadFile
import os
import mimetypes  # Import the mimetypes module
import traceback
import uuid
import json
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_community.document_loaders import (
    TextLoader,  # Import TextFileLoader for .txt files
)
from langchain_core.documents import Document
from app.config import settings
# from app.dependencies import vectorstore

# Configuration is now handled in app.config.settings

LOADER_MAPPING = {
    ".txt": (TextLoader, {}),  # Support .txt files
}


SUPPORTED_MIME_TYPES = [
    "text/plain",  # Support text/plain MIME type for .txt files
    "application/json",  # Support JSON files for intake forms
]

async def process_documents(directory: str):
    """Enhanced document processing with intake forms and specialized headers."""
    
    # Initialize the text splitter for smart chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    all_documents = []
    
    # Iterate through patient directories
    for patient_id in os.listdir(directory):
        patient_dir_path = os.path.join(directory, patient_id)
        
        # Skip if not a directory
        if not os.path.isdir(patient_dir_path):
            print(f"Skipping non-directory: {patient_dir_path}")
            continue
            
        print(f"Processing patient directory: {patient_id}")
        
        # Process files in the patient directory
        for filename in os.listdir(patient_dir_path):
            file_path = os.path.join(patient_dir_path, filename)
            
            # Skip subdirectories, only process files
            if os.path.isdir(file_path):
                print(f"Skipping subdirectory: {file_path}")
                continue
            
            file_extension = os.path.splitext(filename)[1].lower()
            print(f"Examining file: {file_path}")
            
            # Process intake.json files
            if filename.lower() == "intake.json":
                print(f"Processing intake form: {filename}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        intake_data = json.load(f)
                    
                    # Convert to markdown
                    markdown_content = serialize_intake_to_markdown(intake_data)
                    
                    # Create document
                    doc = Document(
                        page_content=markdown_content,
                        metadata={
                            "source": filename,
                            "patient_id": patient_id,
                            "document_type": "intake_form"
                        }
                    )
                    all_documents.append(doc)
                    
                except Exception as e:
                    print(f"Error processing intake file {file_path}: {e}")
                    continue
            
            # Process .txt files (session transcripts)
            elif file_extension == ".txt":
                print(f"Processing text file: {filename}")
                mime_type = mimetypes.guess_type(file_path)[0]
                if mime_type not in SUPPORTED_MIME_TYPES:
                    print(f"Skipping unsupported MIME type: {filename} (MIME type: {mime_type})")
                    continue

                loader_class, loader_args = LOADER_MAPPING.get(file_extension, (None, None))
                if loader_class:
                    try:
                        loader = loader_class(file_path, **loader_args)
                        documents = loader.load()

                        # Add metadata including patient_id and source filename
                        for doc in documents:
                            doc.metadata["source"] = filename
                            doc.metadata["patient_id"] = patient_id
                            doc.metadata["document_type"] = "session_transcript"

                        all_documents.extend(documents)
                    except Exception as e:
                        print(f"Error processing text file {file_path}: {e}")
                        continue
                else:
                    print(f"No loader found for file type: {file_extension}")
            else:
                print(f"Skipping unsupported file: {filename}")

    # Apply smart chunking to all documents
    print(f"Applying smart chunking to {len(all_documents)} documents...")
    
    # Group chunks by document for proper numbering
    all_processed_chunks = []
    
    for doc in all_documents:
        # Split each document individually to maintain proper chunk numbering
        chunks = splitter.split_documents([doc])
        total_chunks = len(chunks)
        
        # Process each chunk with proper numbering
        for chunk_num, chunk in enumerate(chunks, 1):
            # Create specialized header
            header = create_document_header(
                chunk.metadata["source"],
                chunk.metadata["patient_id"],
                chunk_num,
                total_chunks
            )
            
            # Prepend header to content
            new_content = header + chunk.page_content
            
            # Copy existing metadata and add chunk header info
            new_metadata = dict(chunk.metadata)
            new_metadata['chunk_header'] = header.strip()
            new_metadata['chunk_id'] = str(uuid.uuid4())
            new_metadata['chunk_number'] = chunk_num
            new_metadata['total_chunks'] = total_chunks
            
            # Create new document with enhanced content and metadata
            processed_chunk = Document(page_content=new_content, metadata=new_metadata)
            all_processed_chunks.append(processed_chunk)
    
    print(f"Created {len(all_processed_chunks)} processed chunks with specialized headers")
    return all_processed_chunks


def serialize_intake_to_markdown(intake_data: dict) -> str:
    """Convert intake JSON to markdown format without question IDs."""
    markdown_content = []
    
    if "responses" in intake_data:
        for response in intake_data["responses"]:
            title = response.get("title", "")
            content = response.get("response", "")
            
            # Add title as header
            markdown_content.append(f"## {title}")
            markdown_content.append("")  # Empty line
            
            # Add content
            markdown_content.append(content)
            markdown_content.append("")  # Empty line between sections
    
    return "\n".join(markdown_content)


def create_document_header(filename: str, patient_id: str, chunk_num: int, total_chunks: int) -> str:
    """Create appropriate header based on document type."""
    base_name = os.path.splitext(filename)[0].lower()
    
    if base_name == "intake":
        return f"INTAKE FORM - PART {chunk_num} of {total_chunks} — Patient: {patient_id}\n\n"
    elif base_name.startswith("session"):
        session_name = base_name.replace("_", " ").upper()
        return f"{session_name} TRANSCRIPT - PART {chunk_num} of {total_chunks} — Patient: {patient_id}\n\n"
    else:
        # Default header for other files
        return f"Source: {filename} - PART {chunk_num} of {total_chunks} — Patient: {patient_id}\n\n"


async def add_document_to_chroma(
    content: str,
    filename: str,
    patient_id: str,
    document_type: str,
    metadata: Optional[dict] = None
) -> dict:
    """
    Add a single document to the Chroma database.
    
    Args:
        content: The text content of the document
        filename: Name of the file/document
        patient_id: ID of the patient this document belongs to
        document_type: Type of document (e.g., 'session_transcript', 'intake_form', 'assessment', 'note')
        metadata: Optional additional metadata
    
    Returns:
        dict: Status and information about the added document
    """
    try:
        from langchain_chroma import Chroma
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        # Initialize embeddings and vectorstore
        embeddings = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL)
        vectorstore = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            persist_directory=str(settings.chroma_db_path),
            embedding_function=embeddings
        )
        
        # Initialize text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )
        
        # Prepare base metadata
        base_metadata = {
            "source": filename,
            "patient_id": patient_id,
            "document_type": document_type
        }
        
        # Add any additional metadata
        if metadata:
            base_metadata.update(metadata)
        
        # Create document
        doc = Document(page_content=content, metadata=base_metadata)
        
        # Split document into chunks
        chunks = splitter.split_documents([doc])
        total_chunks = len(chunks)
        
        # Process each chunk with headers and metadata
        processed_chunks = []
        for chunk_num, chunk in enumerate(chunks, 1):
            # Create specialized header
            header = create_document_header(filename, patient_id, chunk_num, total_chunks)
            
            # Prepend header to content
            new_content = header + chunk.page_content
            
            # Copy existing metadata and add chunk-specific info
            new_metadata = dict(chunk.metadata)
            new_metadata.update({
                'chunk_header': header.strip(),
                'chunk_id': str(uuid.uuid4()),
                'chunk_number': chunk_num,
                'total_chunks': total_chunks
            })
            
            # Create new document with enhanced content and metadata
            processed_chunk = Document(page_content=new_content, metadata=new_metadata)
            processed_chunks.append(processed_chunk)
        
        # Generate unique IDs for chunks
        chunk_ids = [str(uuid.uuid4()) for _ in processed_chunks]
        
        # Add chunks to vectorstore
        vectorstore.add_documents(documents=processed_chunks, ids=chunk_ids)
        
        return {
            "status": "success",
            "message": f"Successfully added document '{filename}' to database",
            "chunks_created": total_chunks,
            "patient_id": patient_id,
            "document_type": document_type,
            "chunk_ids": chunk_ids
        }
        
    except ImportError as e:
        return {
            "status": "error",
            "message": f"Missing dependencies: {e}",
            "chunks_created": 0
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Error adding document to database: {str(e)}",
            "chunks_created": 0
        }


if __name__ == "__main__":
    import asyncio
    import sys
    import pickle
    import uuid
    
    if len(sys.argv) != 2:
        print("Usage: python process_docs.py <directory_path>")
        print(f"Example: python process_docs.py {settings.SHARED_DOCS_PATH}")
        print("\nThis will:")
        print("  1. Process documents from the specified directory")
        print("  2. Reset and populate Chroma vector database")
        if settings.IS_USING_BM25:
            print(f"  3. Create and save BM25 index as {settings.BM25_INDEX_PATH}")
        else:
            print("  3. Skip BM25 index creation (IS_USING_BM25 = False)")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    
    # Verify directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        sys.exit(1)
    
    async def main():
        print(f"Processing documents from: {directory_path}")
        try:
            chunks = await process_documents(directory_path)
            print(f"Successfully processed {len(chunks)} document chunks")
            
            # Import required modules for index creation
            try:
                from langchain_chroma import Chroma
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                from langchain_community.retrievers import BM25Retriever
                
                print("\n📊 Creating vector database and BM25 index...")
                
                # Initialize embeddings and vectorstore
                embeddings = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL)
                vectorstore = Chroma(
                    collection_name=settings.CHROMA_COLLECTION_NAME,
                    persist_directory=str(settings.chroma_db_path),
                    embedding_function=embeddings
                )
                
                # Reset and populate Chroma DB
                print("Resetting and populating Chroma database...")
                vectorstore.reset_collection()
                ids = [str(uuid.uuid4()) for _ in chunks]
                vectorstore.add_documents(documents=chunks, ids=ids)
                print(f"✅ Added {len(chunks)} documents to Chroma database")
                
                # Create and save BM25 index only if flag is enabled
                if settings.IS_USING_BM25:
                    print("Creating BM25 index...")
                    bm25_retriever = BM25Retriever.from_documents(chunks)
                    bm25_retriever.k = settings.BM25_K
                    
                    # Save BM25 index and documents
                    bm25_data = {
                        "docs": chunks,
                        "retriever_state": {
                            "k": bm25_retriever.k,
                        }
                    }
                    
                    with open(settings.bm25_index_path, "wb") as f:
                        pickle.dump(bm25_data, f)
                    
                    print(f"✅ BM25 index saved to {settings.BM25_INDEX_PATH}")
                else:
                    print("⏭️  Skipping BM25 index creation (IS_USING_BM25 = False)")
                print("\n🎉 Document processing and indexing complete!")
                print(f"Processed {len(chunks)} chunks from {directory_path}")
                if settings.IS_USING_BM25:
                    print("✅ BM25 index created and saved")
                else:
                    print("⏭️  BM25 index creation skipped")
                
            except ImportError as e:
                print(f"\n⚠️  Warning: Could not create indices due to missing dependencies: {e}")
                print("Document processing completed, but indices were not created.")
                print("Make sure you have langchain packages installed to create indices.")
            
            # Print sample chunks for verification
            print(f"\n📋 Sample processed chunks:")
            for i, chunk in enumerate(chunks[:3]):
                print(f"\nChunk {i+1}:")
                print(f"  Source: {chunk.metadata.get('source', 'Unknown')}")
                print(f"  Patient ID: {chunk.metadata.get('patient_id', 'Unknown')}")
                print(f"  Type: {chunk.metadata.get('document_type', 'Unknown')}")
                print(f"  Content preview: {chunk.page_content[:200]}...")
            
            if len(chunks) > 3:
                print(f"\n... and {len(chunks) - 3} more chunks")
                
        except Exception as e:
            print(f"❌ Error processing documents: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    asyncio.run(main())


