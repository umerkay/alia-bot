# Environment Variables Configuration

This application uses environment variables for configuration. All paths and settings can be customized through environment variables.

## Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your specific values.

## Available Environment Variables

### Google AI Configuration
- `GOOGLE_MODEL`: Google AI model to use (default: `gemini-2.5-flash`)
- `TEMPERATURE`: Model temperature for response generation (default: `0.4`)
- `STREAMING`: Enable streaming responses (default: `true`)

### Path Configuration
- `CHROMA_DB_PATH`: Path to Chroma vector database directory (default: `shared_chroma_db`)
- `SHARED_DOCS_PATH`: Path to shared documents directory (default: `shared_docs`)
- `BM25_INDEX_PATH`: Path to BM25 index pickle file (default: `bm25_index.pkl`)

### Database Configuration
- `CHROMA_COLLECTION_NAME`: Name of the Chroma collection (default: `main_collection`)
- `EMBEDDING_MODEL`: Embedding model to use (default: `models/text-embedding-004`)

### BM25 Configuration
- `IS_USING_BM25`: Enable BM25 search functionality (default: `false`)
- `BM25_K`: Number of results to return from BM25 search (default: `3`)

### Text Processing Configuration
- `CHUNK_SIZE`: Size of text chunks for processing (default: `1000`)
- `CHUNK_OVERLAP`: Overlap between text chunks (default: `200`)

### Retrieval Configuration
- `VECTOR_SEARCH_K`: Number of results to return from vector search (default: `10`)
- `ENSEMBLE_VECTOR_WEIGHT`: Weight for vector search in ensemble retrieval (default: `0.6`)
- `ENSEMBLE_BM25_WEIGHT`: Weight for BM25 search in ensemble retrieval (default: `0.4`)

## Usage Examples

### Processing Documents
```bash
# Default paths (using environment variables)
python app/services/process_docs.py $SHARED_DOCS_PATH

# Or with explicit path
python app/services/process_docs.py /path/to/your/documents
```

### Running the Application
The application will automatically use the configured paths from environment variables.

## Path Structure
With default environment variables, the application expects:
```
your-project/
├── shared_chroma_db/          # Vector database storage
├── shared_docs/               # Source documents
├── bm25_index.pkl            # BM25 index file (if enabled)
└── .env                      # Your environment configuration
```

You can customize any of these paths by setting the appropriate environment variables.
