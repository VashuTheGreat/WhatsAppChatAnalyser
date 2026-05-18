from langchain_groq import ChatGroq
from src.WhatsApp_Analyser.constants import MODEL_NAME,TEMPERATURE
model=ChatGroq(model=MODEL_NAME,temperature=TEMPERATURE)