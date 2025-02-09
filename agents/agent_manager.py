from agents.summarization import summarize_webpage
from agents.lights import control_lights

AGENT_FUNCTIONS = {
    "summarization": summarize_webpage,
    "turn_on_lights": control_lights,
    "turn_off_lights": control_lights,
}

def execute_agent(intent: str, command: str) -> str:
    """Calls the appropriate agent based on detected intent."""
    print(f"🤖 Executing agent: {intent}")
    return AGENT_FUNCTIONS.get(intent, lambda _: "❌ Unknown agent.")(intent, command)

