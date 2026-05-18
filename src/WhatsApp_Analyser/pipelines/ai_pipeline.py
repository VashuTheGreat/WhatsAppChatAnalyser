import logging
from src.WhatsApp_Analyser.graphs.builder import graph
from langchain_core.messages import HumanMessage
from utils.asyncHandler import asyncHandler


class AIPipeline:
    def __init__(self):
        logging.info("Initializing AIPipeline")
        self.graph = graph

    @asyncHandler
    async def get_response(self, message: str, file_path: str, thread_id: str = "default"):
        logging.info(f"AI Pipeline: Processing message for thread {thread_id}")
        
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {
            "messages": [HumanMessage(content=message)],
            "file_path": file_path
        }
        
        res = await self.graph.ainvoke(inputs, config=config)
        ai_msg = res['messages'][-1].content
        
        logging.info("AI Pipeline: Response generated successfully")
        return ai_msg
