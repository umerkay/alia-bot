import json
from app.agent import get_agent_executor, chat_history_store, MAX_MESSAGES
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
import re


async def chat_stream(data: str, conversation_id: str = "abc123"):
    """Handles streaming chat responses from LangGraph workflow."""
    
    # Tool mapper for display names
    tool_mapper = {
        "transcript_agent": "Analyzing patient transcripts",
        "intake_form_retriever": "Retrieving intake form information",
        "ehr_retriever": "Retrieving EHR data",
        "assessment_agent": "Generating patient assessment",
        "rag_tool": "Searching medical knowledge base"
    }
    
    try:
        # Check if agent executor is initialized
        agent_executor = get_agent_executor()
        if agent_executor is None:
            yield chat_response(conversation_id, "Agent not initialized. Please restart the server.", "error")
            return
        
        # Configuration for the workflow
        config = {"configurable": {"thread_id": conversation_id, "patient_id": "patient1"}}
        
        # Get previous messages and add new user message
        previous_messages = chat_history_store.get(conversation_id, [])
        new_user_message = HumanMessage(content=data)
        messages = previous_messages + [new_user_message]
        
        # Track response and final AI message for history
        response_count = 0
        final_ai_message = None
        
        try:
            async for chunk in agent_executor.astream_events({"messages": messages}, config, stream_mode="messages", version="v2"):
                response_count += 1
                kind = chunk["event"]
                
                if kind == "on_chat_model_stream":
                    content = chunk["data"]["chunk"].content
                    if content:
                        yield chat_response(conversation_id, content, "")
                        final_ai_message = chunk["data"]["chunk"]
                
                elif kind == "on_tool_start":
                    tool_name = chunk["name"]
                    if tool_name in tool_mapper:
                        message = tool_mapper.get(tool_name, f"Starting {tool_name}...")
                        yield chat_response(conversation_id, message, tool_name)
                
                elif kind == "on_tool_end":
                    tool_name = chunk["name"]
                    if tool_name in tool_mapper:
                        # Extract and send tool output
                        output_content = chunk["data"].get("output")
                        output_msg = f"Completed {tool_name}"
                        
                        if hasattr(output_content, "content"):
                            output_msg = output_content.content
                        elif isinstance(output_content, str):
                            output_msg = output_content
                        
                        yield chat_response(conversation_id, output_msg, tool_name)
            
            # Update chat history with final AI response
            if final_ai_message:
                chat_history_store[conversation_id] = (
                    messages + [final_ai_message]
                )[-MAX_MESSAGES:]
            
            if response_count == 0:
                yield chat_response(conversation_id, "No response from workflow. There might be an issue with the agent configuration.", "error")
            
        except Exception as workflow_error:
            print(f"Workflow error: {workflow_error}")
            yield chat_response(conversation_id, f"Workflow error: {str(workflow_error)}", "error")
        
        # Send final stop signal
        yield chat_response(conversation_id, "", "", "stop")
        
    except Exception as e:
        print(f"Error in chat_stream: {e}")
        yield chat_response(conversation_id, f"Error: {str(e)}", "error")


def chat_response(conversation_id, content, tool, finish_reason=None):
    response = {
        "conversation_id": conversation_id,
        "object": "chat.completion.chunk",
        "created": str(datetime.now().timestamp()),
        "model": "gemini-2.0-flash",
        "system_fingerprint": "alia-ai",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": content,
                    "tool": tool
                },
                "logprobs": None,
                "finish_reason": finish_reason
            }
        ]
    }
    
    return "data: " + json.dumps(response) + "\n\n"
