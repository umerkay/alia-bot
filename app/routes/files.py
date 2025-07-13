from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pathlib import Path
from typing import Optional
import json
from app.services.process_docs import add_document_to_chroma
from app.config import settings

router = APIRouter(prefix="/files", tags=["files"])

@router.get("/json/{file_name}", response_model=dict)
async def get_json_file(file_name: str):
    file_path = settings.shared_docs_path / file_name
    if not file_path.exists() or file_path.suffix != ".json":
        raise HTTPException(status_code=404, detail="File not found")
    with file_path.open("r") as file:
        return json.load(file)

@router.post("/add-document/{patient_id}")
async def add_document(
    patient_id: str,
    document_type: str = Form(...),
    filename: str = Form(...),
    content: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Add a new document to the Chroma database.
    
    Parameters:
    - patient_id: ID of the patient this document belongs to
    - document_type: Type of document (e.g., 'session_transcript', 'intake_form', 'assessment', 'note')
    - filename: Name of the document
    - content: Text content (if not uploading a file)
    - file: File to upload (if not providing content directly)
    """
    try:
        # Get content either from form field or uploaded file
        if file is not None:
            file_content = await file.read()
            
            if file.content_type == "text/plain":
                text_content = file_content.decode('utf-8')
            elif file.content_type == "application/json":
                json_data = json.loads(file_content.decode('utf-8'))
                if document_type == "intake_form":
                    from app.services.process_docs import serialize_intake_to_markdown
                    text_content = serialize_intake_to_markdown(json_data)
                else:
                    text_content = json.dumps(json_data, indent=2)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
            
            actual_filename = file.filename or filename
            metadata = {"uploaded_filename": file.filename, "content_type": file.content_type}
            
        elif content is not None:
            text_content = content
            actual_filename = filename
            metadata = {"input_method": "text"}
            
        else:
            raise HTTPException(status_code=400, detail="Either 'content' or 'file' must be provided")
        
        # Add to database
        result = await add_document_to_chroma(
            content=text_content,
            filename=actual_filename,
            patient_id=patient_id,
            document_type=document_type,
            metadata=metadata
        )
        
        if result["status"] == "success":
            return {
                "message": result["message"],
                "chunks_created": result["chunks_created"],
                "patient_id": result["patient_id"],
                "document_type": result["document_type"]
            }
        else:
            raise HTTPException(status_code=500, detail=result["message"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")
