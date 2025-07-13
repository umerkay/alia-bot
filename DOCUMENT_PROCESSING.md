# Document Processing Workflow

This project separates document processing concerns into two distinct phases:

## 1. Document Processing Phase

Process documents and create indices by running the process_docs module as a script:

```bash
# Process documents and create indices (from project root)
python app/services/process_docs.py shared_docs
```

This will:
- Process all documents in the specified directory
- Reset and populate the Chroma vector database
- Create and save a BM25 index as `bm25_index.pkl`

## 2. Runtime Phase

The RAG tool (`rag_tool.py`) loads existing data instead of processing documents:

- Loads the existing Chroma database from `shared_chroma_db/`
- Loads the BM25 index from `bm25_index.pkl`
- Creates ensemble retrievers for semantic search

## File Structure

```
project/
├── bm25_index.pkl                 # Generated BM25 index (after processing)
├── shared_chroma_db/              # Generated Chroma database (after processing)
├── shared_docs/                   # Source documents
│   ├── patient1/
│   └── patient2/
└── app/
    └── services/
        ├── process_docs.py        # Document processing + index creation
        └── rag_tool.py           # RAG retrieval (loads existing data)
```

## Benefits of This Separation

1. **Performance**: Document processing happens once, not on every application start
2. **Reliability**: Clear separation between data preparation and runtime
3. **Flexibility**: Can re-process documents independently of the main application
4. **Development**: Easier to test and debug document processing logic

## Usage Examples

```bash
# Full workflow (from project root)
python app/services/process_docs.py shared_docs

# Then your main application can use the generated indices
python -m app.main
```
