from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from app.config import settings
from app.services.rag_tool import create_intake_form_retrieval_tool, create_transcript_retrieval_tool, create_assessment_retrieval_tool
from app.services.graph_rag_tool import create_ehr_retrieval_tool
from app.services.prompts import get_main_agent_prompt, get_transcript_agent_prompt, get_assessment_agent_prompt
from typing import Dict, List
import datetime
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

# Setup LangChain components
model = ChatGoogleGenerativeAI(
    model=settings.GOOGLE_MODEL, 
    temperature=settings.TEMPERATURE, 
)

chat_history_store: Dict[str, List[BaseMessage]] = {}
MAX_MESSAGES = 20  # 10 turns (Human + AI)

# Global agent executor
agent_executor = None

async def build_main_agent():
    """Build the main clinical agent that routes queries"""
    prompt = get_main_agent_prompt()
    tools = [create_intake_form_retrieval_tool(), create_ehr_retrieval_tool(), get_transcript_agent_tool(), get_assessment_agent_tool()]
    return create_react_agent(model, tools, prompt=prompt)

async def build_transcript_agent():
    """Build the specialized transcript agent"""
    tools = [create_transcript_retrieval_tool()]
    prompt = get_transcript_agent_prompt()
    return create_react_agent(model, tools, prompt=prompt)

_transcript_agent = None

async def get_transcript_agent():
    global _transcript_agent
    if _transcript_agent is None:
        _transcript_agent = await build_transcript_agent()
    return _transcript_agent

def get_transcript_agent_tool():
    @tool
    async def transcript_agent(query: str, config: RunnableConfig) -> str:
        """Use this tool for transcript-related questions.
        
        Args:
            query (str): The query to ask the transcript agent.

        Returns:
            str: The response from the transcript agent.

        """
        print(f"Calling transcript agent with query: {query}")
        transcript_agent = await get_transcript_agent()
        response = await transcript_agent.ainvoke({"messages": [HumanMessage(content=query)]}, config=config)
        last_message = response.get("messages", [])[-1] if response.get("messages") else None
        return last_message.content.strip() if last_message else "(No response from transcript agent)"
    
    return transcript_agent

async def build_assessment_agent():
    """Build the specialized assessment agent"""
    tools = [create_assessment_retrieval_tool()]
    prompt = get_assessment_agent_prompt()
    return create_react_agent(model, tools, prompt=prompt)

_assessment_agent = None

async def get_assessment_agent():
    global _assessment_agent
    if _assessment_agent is None:
        _assessment_agent = await build_assessment_agent()
    return _assessment_agent

def get_assessment_agent_tool():
    @tool
    async def assessment_agent(query: str, config: RunnableConfig) -> str:
        """Use this tool for assessment-related questions.
        
        Args:
            query (str): The query to ask the assessment agent.

        Returns:
            str: The response from the assessment agent.

        """
        print(f"Calling assessment agent with query: {query}")
        assessment_agent = await get_assessment_agent()
        response = await assessment_agent.ainvoke({"messages": [HumanMessage(content=query)]}, config=config)
        last_message = response.get("messages", [])[-1] if response.get("messages") else None
        return last_message.content.strip() if last_message else "(No response from assessment agent)"
    
    return assessment_agent

def get_agent_executor():
    return agent_executor

async def init_agent():
    """Initialize the global agent executor"""
    global agent_executor
    try:
        agent_executor = await build_main_agent()
        print("Agent initialized successfully")
            
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        import traceback
        traceback.print_exc()
