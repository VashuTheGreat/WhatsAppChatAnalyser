import logging
from src.WhatsApp_Analyser.models.state_model import State
from utils.asyncHandler import asyncHandler
from src.WhatsApp_Analyser.tools.CodeRunner_tool import code_runner
from src.WhatsApp_Analyser.llm import model as base_model
from src.WhatsApp_Analyser.prompts import CHAT_LLM_PROMPT
from langchain_core.messages import SystemMessage

tools = [code_runner]

@asyncHandler
async def chat(state: State):
    logging.info("Chat node initiated")
    
    messages = state.messages
    if not any(isinstance(m, SystemMessage) for m in messages):
        system_content = f"{CHAT_LLM_PROMPT}\n\nAvailable data file path: {state.file_path}"
        messages = [SystemMessage(content=system_content)] + messages
    
    llm_with_tools = base_model.bind_tools(tools=tools)
    logging.info(f"Invoking LLM with tools: {[t.__name__ for t in tools]}")
    response = await llm_with_tools.ainvoke(messages)
    
    return {"messages": [response]}
