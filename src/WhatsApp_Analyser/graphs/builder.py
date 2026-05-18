import logging
from langgraph.graph import START, END, StateGraph
from src.WhatsApp_Analyser.models.state_model import State
from src.WhatsApp_Analyser.nodes.chat_node import chat as chat_node, tools
from langgraph.prebuilt import ToolNode, tools_condition
from src.WhatsApp_Analyser.memory import memory

logging.info("Initializing graph builder")

workflow = StateGraph(State)

workflow.add_node("chat", chat_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "chat")
workflow.add_conditional_edges(
    "chat",
    tools_condition,
)
workflow.add_edge("tools", "chat")

graph = workflow.compile(checkpointer=memory)

try:
    with open("graph.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())
    logging.info("Graph PNG diagram saved")
except Exception as e:
    logging.error(f"Failed to save graph diagram: {e}")


