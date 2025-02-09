import os
import json
import langid
import tempfile
import requests
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from dotenv import load_dotenv

load_dotenv()

AI_ASSISTANT_PROMPT = os.getenv("AI_ASSISTANT_PROMPT")
ANYTHING_LLM_API_URL = os.getenv("ANYTHING_LLM_API_URL")
ANYTHING_LLM_API_KEY = os.getenv("ANYTHING_LLM_API_KEY")
ANYTHING_LLM_API_CHAT_URL = f"{ANYTHING_LLM_API_URL}/v1/openai/chat/completions"
ANYTHING_LLM_CHAT_WORKSPACE = os.getenv("ANYTHING_LLM_CHAT_WORKSPACE")
ANYTHING_LLM_CHAT_TEMPERATURE = os.getenv("ANYTHING_LLM_CHAT_TEMPERATURE")

FIRST_LANGUAGE_NAME = os.getenv("FIRST_LANGUAGE_NAME")
FIRST_LANGUAGE_CODE = os.getenv("FIRST_LANGUAGE_CODE")
SECOND_LANGUAGE_NAME = os.getenv("SECOND_LANGUAGE_NAME")
SECOND_LANGUAGE_CODE = os.getenv("SECOND_LANGUAGE_CODE")
SECOND_LANGUAGE_SPOKEN_CODE = os.getenv("SECOND_LANGUAGE_SPOKEN_CODE")

DEBUG_AI_RESPONSE = os.getenv("DEBUG_AI_RESPONSE", "false").lower() == "true"

def detect_prompt_language(text: str) -> str:
    detected_lang = langid.classify(text)[0]
    print(f"🔍 Detected prompt language: {detected_lang}")
    return SECOND_LANGUAGE_NAME if detected_lang == SECOND_LANGUAGE_CODE else FIRST_LANGUAGE_NAME

def detect_spoken_language(text: str) -> str:
    detected_lang = langid.classify(text)[0]
    if DEBUG_AI_RESPONSE == "true":
        print(f"🔍 Detected spoken language: {detected_lang}")
    return SECOND_LANGUAGE_SPOKEN_CODE if detected_lang == SECOND_LANGUAGE_CODE else FIRST_LANGUAGE_CODE

def send_command_to_ai(command: str):
    lang = detect_prompt_language(command)
    system_prompt = f"{AI_ASSISTANT_PROMPT}\n\nResponse only in {lang}."

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": command}
        ],
        "model": ANYTHING_LLM_CHAT_WORKSPACE,
        "temperature": ANYTHING_LLM_CHAT_TEMPERATURE,
        "stream": False,
    }

    if DEBUG_AI_RESPONSE == "true": 
        print(f"🤖 Sending to chat AI: {payload}")

    headers = {
        "Authorization": f"Bearer {ANYTHING_LLM_API_KEY}",
        "Content-Type": "application/json",
        "accept": "*/*"
    }

    response = requests.post(ANYTHING_LLM_API_CHAT_URL, headers=headers, data=json.dumps(payload))

    if DEBUG_AI_RESPONSE == "true":
        print(f"🤖 AI Response: {response.text}")

    if response.status_code == 200:
        ai_response = response.json()["choices"][0]["message"]["content"]
        print(f"📝 AI Response: {ai_response}")
        return ai_response
    else:
        print(f"❌ Error communicating with AI: {response.status_code}")
        return None

def stream_voice_response(text: str):
    lang = detect_spoken_language(text)
    print(f"🎤 Speaking AI response in {lang}...")

    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmp_audio:
            tts.save(tmp_audio.name)
            audio = AudioSegment.from_file(tmp_audio.name, format="mp3")
            play(audio)
            print(f"🎤 Speaking ended...")
    except Exception as e:
        print(f"❌ Failed to generate speech: {e}")