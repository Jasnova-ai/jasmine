import os
import re
import langgraph
import requests
from bs4 import BeautifulSoup
from langgraph.graph import StateGraph
from pydantic import BaseModel
from typing import Dict, Any
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

AGENT_API_URL = os.getenv("AGENT_API_URL")
AGENT_MODEL = os.getenv("AGENT_MODEL")
DEBUG_AI_RESPONSE = os.getenv("DEBUG_AI_RESPONSE", "false").lower() == "true"

class SummarizationState(BaseModel):
    url: str
    html: str = ""
    text: str = ""
    summary: str = ""

def fetch_webpage(state: SummarizationState) -> Dict[str, Any]:
    response = requests.get(state.url)
    return {"html": response.text} if response.status_code == 200 else {}

def extract_text(state: SummarizationState) -> Dict[str, Any]:
    soup = BeautifulSoup(state.html, 'html.parser')
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()
    text = ' '.join(soup.get_text(separator=' ').split())
    return {"text": text[:5000]}

def summarization(state: SummarizationState) -> Dict[str, Any]:
    payload = {
        "model": AGENT_MODEL,
        "prompt": f"Summarize this:\n\n{state.text}",
        "max_length": 500
    }

    if DEBUG_AI_RESPONSE == "true":
        print(f"🤖 Sending to agent API: {payload}")

    response = requests.post(f"{AGENT_API_URL}/completions", json=payload)
    return {"summary": response.json().get("choices", [{}])[0].get("text", "No summary found.")}

graph = StateGraph(SummarizationState)
graph.add_node("fetch_webpage", fetch_webpage)
graph.add_node("extract_text", extract_text)
graph.add_node("summarize_text", summarization)
graph.add_edge("fetch_webpage", "extract_text")
graph.add_edge("extract_text", "summarize_text")
graph.set_entry_point("fetch_webpage")

summarization_pipeline = graph.compile()

def summarize_webpage(intent: str, command: str) -> str:
    """Executes the summarization agent and extracts a valid URL."""

    # Regular expression to capture a full URL or just a domain
    url_match = re.search(r"(https?://\S+|www\.\S+|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b)", command)
    
    if url_match:
        url = url_match.group(0)
        
        # Ensure URL has a scheme (http/https), add https:// if missing
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            url = f"https://{url}"
        
        # Call the summarization agent
        summary = summarization_pipeline.invoke({"url": url}).get("summary", "⚠️ No summary available.")
    else:
        summary = "❌ No valid URL found."

    print(f"📝 Summary: {summary}")
    return summary

