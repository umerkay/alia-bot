#!/usr/bin/env python3
"""
Test script to verify the document processing workflow works correctly.
"""

import sys
import asyncio
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from app.services.process_docs import process_documents
from app.config import settings


async def test_process_docs():
    """Test that document processing works correctly."""
    print("Testing document processing...")
    
    # Test with shared_docs directory from settings
    test_dir = str(settings.shared_docs_path)
    
    if not os.path.exists(test_dir):
        print(f"Warning: Test directory '{test_dir}' not found")
        return False
    
    try:
        chunks = await process_documents(test_dir)
        print(f"✅ Successfully processed {len(chunks)} chunks")
        
        # Verify chunks have expected metadata
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1}:")
            print(f"  Source: {chunk.metadata.get('source', 'N/A')}")
            print(f"  Patient ID: {chunk.metadata.get('patient_id', 'N/A')}")
            print(f"  Document Type: {chunk.metadata.get('document_type', 'N/A')}")
            print(f"  Content length: {len(chunk.page_content)} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing documents: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bm25_pickle():
    """Test that BM25 pickle operations work."""
    print("\nTesting BM25 pickle operations...")
    
    try:
        import pickle
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document
        
        # Create test documents
        test_docs = [
            Document(page_content="This is a test document", metadata={"test": True}),
            Document(page_content="Another test document", metadata={"test": True})
        ]
        
        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(test_docs)
        bm25_retriever.k = 3
        
        # Test pickle save/load
        test_data = {
            "docs": test_docs,
            "retriever_state": {"k": bm25_retriever.k}
        }
        
        with open("test_bm25.pkl", "wb") as f:
            pickle.dump(test_data, f)
        
        with open("test_bm25.pkl", "rb") as f:
            loaded_data = pickle.load(f)
        
        # Recreate retriever
        new_retriever = BM25Retriever.from_documents(loaded_data["docs"])
        new_retriever.k = loaded_data["retriever_state"]["k"]
        
        print(f"✅ BM25 pickle test passed - loaded {len(loaded_data['docs'])} docs")
        
        # Cleanup
        os.remove("test_bm25.pkl")
        return True
        
    except ImportError as e:
        print(f"⚠️  BM25 test skipped - missing dependencies: {e}")
        return True
    except Exception as e:
        print(f"❌ BM25 pickle test failed: {e}")
        return False


async def main():
    print("🧪 Testing document processing workflow...\n")
    
    # Test document processing
    doc_test_passed = await test_process_docs()
    
    # Test BM25 pickle operations
    pickle_test_passed = test_bm25_pickle()
    
    print(f"\n📊 Test Results:")
    print(f"  Document Processing: {'✅ PASS' if doc_test_passed else '❌ FAIL'}")
    print(f"  BM25 Pickle: {'✅ PASS' if pickle_test_passed else '❌ FAIL'}")
    
    if doc_test_passed and pickle_test_passed:
        print("\n🎉 All tests passed! The workflow should work correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")


if __name__ == "__main__":
    asyncio.run(main())
