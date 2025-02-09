import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

INTENT_CLASSIFICATION_PROMPT = os.getenv("INTENT_CLASSIFICATION_PROMPT")

ANYTHING_LLM_API_URL = os.getenv("ANYTHING_LLM_API_URL")
ANYTHING_LLM_API_KEY = os.getenv("ANYTHING_LLM_API_KEY")
ANYTHING_LLM_API_CHAT_URL = f"{ANYTHING_LLM_API_URL}/v1/openai/chat/completions"

ANYTHING_LLM_INTENT_WORKSPACE = os.getenv("ANYTHING_LLM_INTENT_WORKSPACE")
ANYTHING_LLM_INTENT_TEMPERATURE = os.getenv("ANYTHING_LLM_INTENT_TEMPERATURE")

DEBUG_AI_RESPONSE = os.getenv("DEBUG_AI_RESPONSE", "false").lower() == "true"

def classify_intent(command: str) -> str:
    prompt_template = INTENT_CLASSIFICATION_PROMPT.strip()
    prompt = prompt_template.format(command=command) 

    messages = [
        {"role": "user", "content": prompt},
    ]

    payload = {
        "messages": messages,
        "model": ANYTHING_LLM_INTENT_WORKSPACE,
        "temperature": ANYTHING_LLM_INTENT_TEMPERATURE,
    }

    if DEBUG_AI_RESPONSE == "true":
        print(f"🤖 Sending to intent AI: {payload}")

    headers = {
        "Authorization": f"Bearer {ANYTHING_LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(ANYTHING_LLM_API_CHAT_URL, headers=headers, data=json.dumps(payload))

        # Handle empty or non-JSON responses
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return "llm"

        try:
            json_response = response.json()
            return json_response["choices"][0]["message"]["content"].strip().lower()
        except json.JSONDecodeError:
            print(f"❌ JSON Decode Error: Response text = {response.text}")
            return "llm"

    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {e}")
        return "llm"
