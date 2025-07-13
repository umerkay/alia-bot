import json
from app.agent import get_agent_executor, chat_history_store, MAX_MESSAGES, ClinicalAgentState, create_initial_clinical_state
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
import re


async def chat_stream(data: str, conversation_id: str = "abc123"):
    """Handles streaming chat responses from LangGraph workflow."""
    
    try:
        # Check if agent executor is initialized
        agent_executor = get_agent_executor()
        if agent_executor is None:
            yield chat_response(conversation_id, "Agent not initialized. Please restart the server.", "error")
            return
        
        # Configuration for the workflow
        config = {"configurable": {"thread_id": conversation_id, "patient_id": "patient2"}}
        
        # Create initial state for the clinical workflow
        previous_messages = chat_history_store.get(conversation_id, [])
        new_user_message = HumanMessage(content=data)
        messages = previous_messages + [new_user_message]
        
        # Build the initial state using the ClinicalAgentState helper
        initial_state = create_initial_clinical_state(messages)
        
        # Track response and final AI message for history
        response_count = 0
        final_ai_message = None
        
        try:
            async for chunk in agent_executor.astream(initial_state, config):
                response_count += 1
                
                # Process each node output in the chunk
                for node_output in chunk.values():
                    # Get the last message from this node
                    node_messages = node_output.get("messages", [])
                    if not node_messages:
                        continue
                    
                    last_message = node_messages[-1]
                    if not hasattr(last_message, 'content') or not last_message.content:
                        continue
                    
                    content = last_message.content.strip()
                    if not content:
                        continue
                    
                    # Determine message type and stream accordingly
                    if "TRANSCRIPT_NEEDED:" in content:
                        yield chat_response(conversation_id, "Routing to transcript analysis...", "routing")
                    # elif "Analyzing transcripts" in content or node_output.get("query_type") == "transcript":
                        # yield chat_response(conversation_id, "Analyzing transcripts...", "transcript_processing")
                    else:
                        # This is the actual response content
                        yield chat_response(conversation_id, content, "")
                        final_ai_message = last_message
            
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
        "system_fingerprint": "agro-ai",
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
