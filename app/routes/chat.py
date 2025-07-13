from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi import Form
import json
import uuid
from datetime import datetime
from app.services.chat_service import chat_stream
from app.config import settings

router = APIRouter()

@router.post("/chat")
async def chat(request: Request):
    try:
        # Parse the request body
        body = await request.json()
        conversation_id = body.get("conversation_id")
        prompt = body.get("prompt")
        
        # Validate required fields
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        
        # Generate conversation_id if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Create an async generator that yields the existing chat response format
        async def event_generator():
            try:
                # Use the chat_stream function with streaming enabled
                async for response in chat_stream(prompt, conversation_id):
                    # The chat_stream already returns the proper format with "data: " prefix
                    yield response
            
            except Exception as e:
                # Handle any errors during streaming - maintain existing format
                error_response = {
                    "conversation_id": conversation_id,
                    "object": "chat.completion.chunk",
                    "created": str(datetime.now().timestamp()),
                    "model": settings.GOOGLE_MODEL,
                    "system_fingerprint": "alia-ai",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": f"Error: {str(e)}",
                                "tool": "error"
                            },
                            "logprobs": None,
                            "finish_reason": "error"
                        }
                    ]
                }
                yield f"data: {json.dumps(error_response)}\n\n"
        
        # Return a StreamingResponse with the event generator
        return StreamingResponse(
            event_generator(), 
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable buffering for Nginx
            }
        )
    
    except Exception as e:
        # Handle any errors in parsing the request
        raise HTTPException(
            status_code=400, 
            detail=f"Error processing request: {str(e)}"
        )

# @router.get("/new_chat")
# def new_chat():
#     new_uuid = str(uuid.uuid4())
#     return {"conversation_id": new_uuid}
