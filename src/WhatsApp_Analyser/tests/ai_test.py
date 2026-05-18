import sys
import os
import asyncio
from langchain_core.messages import HumanMessage


sys.path.append(os.getcwd())

from dotenv import load_dotenv
load_dotenv()
from logger import *
import logging

from src.WhatsApp_Analyser.graphs.builder import graph

async def main():
    logging.info("Starting AI test")
    config = {"configurable": {"thread_id": "1"}}
    inputs = {
        "messages": [HumanMessage(content="how many students are there")],
        "file_path": "WhatsAppChatAnalyser/artifacts/ingestion/features/data.csv"
    }
    
    try:
        res = await graph.ainvoke(inputs, config=config)
        logging.info("Graph invocation completed")
        print(res['messages'][-1].content)
    except Exception as e:
        logging.error(f"Error during graph invocation: {e}")

if __name__ == "__main__":
    asyncio.run(main())