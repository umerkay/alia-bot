# Document Addition API

This document describes the new functionality for adding documents to the Chroma database via API endpoint.

## Overview

A new API endpoint has been added to allow real-time addition of documents to the Chroma vector database. The endpoint supports both text content input and file uploads.

## API Endpoint

**Endpoint:** `POST /files/add-document/{patient_id}`

**Description:** Add a new document to the Chroma database using either text content or file upload.

**Parameters:**
- `patient_id` (path): ID of the patient this document belongs to
- `document_type` (form field): Type of document (e.g., 'session_transcript', 'intake_form', 'assessment', 'note', 'progress_note')
- `filename` (form field): Name of the document
- `content` (form field, optional): Text content of the document (if not uploading file)
- `file` (form field, optional): File to upload (if not providing text content)

**Note:** Either `content` or `file` must be provided, but not both.

**Supported File Types:**
- Text files (`.txt`) with MIME type `text/plain`
- JSON files (`.json`) with MIME type `application/json`

**Response:**
```json
{
  "message": "Successfully added document to database",
  "chunks_created": 3,
  "patient_id": "patient1",
  "document_type": "session_transcript"
}
```

## Document Types

The system supports the following document types:

- `session_transcript` - Therapy session transcripts
- `intake_form` - Patient intake forms
- `assessment` - Clinical assessments
- `progress_note` - Progress notes
- `note` - General notes
- Custom types can be added as needed

## Document Processing

All documents are processed as follows:

1. **Text Splitting**: Documents are split into chunks using RecursiveCharacterTextSplitter
   - Chunk size: 1000 characters
   - Overlap: 200 characters

2. **Header Addition**: Each chunk gets a specialized header containing:
   - Document type and chunk information
   - Patient ID
   - Source filename

3. **Metadata Enhancement**: Each chunk includes:
   - Source filename
   - Patient ID
   - Document type
   - Chunk number and total chunks
   - Unique chunk ID
   - Custom metadata (if provided)

4. **Vector Embedding**: Chunks are embedded using Google's text-embedding-004 model

5. **Database Storage**: Embedded chunks are stored in the Chroma vector database

## Usage Examples

### Using cURL

Add text content:
```bash
curl -X POST "http://localhost:8000/files/add-document/patient1" \
  -F "document_type=progress_note" \
  -F "filename=progress_note.txt" \
  -F "content=Patient showed improvement in mood and reported better sleep quality."
```

Upload a file:
```bash
curl -X POST "http://localhost:8000/files/add-document/patient1" \
  -F "document_type=session_transcript" \
  -F "filename=session.txt" \
  -F "file=@session_transcript.txt"
```

### Using Python

```python
import requests

# Add text content
response = requests.post(
    "http://localhost:8000/files/add-document/patient1",
    data={
        "document_type": "note",
        "filename": "test.txt",
        "content": "Document content here..."
    }
)

# Upload file
with open("document.txt", "rb") as f:
    response = requests.post(
        "http://localhost:8000/files/add-document/patient1",
        files={"file": f},
        data={
            "document_type": "session_transcript",
            "filename": "uploaded_doc.txt"
        }
    )
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200` - Success
- `400` - Bad request (invalid parameters, unsupported file type)
- `500` - Server error (database issues, processing errors)

Error responses include a detailed error message in the `detail` field.
