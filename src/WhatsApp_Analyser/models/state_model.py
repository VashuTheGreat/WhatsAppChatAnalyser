import logging
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from typing import Annotated, List, Optional
import operator

class State(BaseModel):
    messages: Annotated[List[BaseMessage], operator.add]
    file_path: Optional[str] = None