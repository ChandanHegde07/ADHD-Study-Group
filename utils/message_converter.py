from typing import List, Dict, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

def get_langchain_messages_from_st_history(st_messages: List[Dict[str, str]]) -> List[BaseMessage]:
    langchain_messages = []
    for msg in st_messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))
    return langchain_messages