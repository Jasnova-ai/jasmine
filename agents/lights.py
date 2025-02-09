import os
from langgraph.graph import StateGraph
from pydantic import BaseModel
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DEBUG_AI_RESPONSE = os.getenv("DEBUG_AI_RESPONSE", "false").lower() == "true"

# Global state to keep track of lights
light_state = None

class LightControlState(BaseModel):
    status: str = "off"
    response: str = ""

def light_status(state: LightControlState) -> Dict[str, Any]:
    return {"status": state.status, "response": f"The lights are currently {state.status.upper()}."}

def turn_on_lights(state: LightControlState) -> Dict[str, Any]:
    if state.status == "off":
        print("💡 Turning on the lights...")
        return {"status": "on", "response": "Lights are ON now."}
    return {"status": "on", "response": "Lights are already ON."}

def turn_off_lights(state: LightControlState) -> Dict[str, Any]:
    if state.status == "on":
        print("💡 Turning off the lights...")
        return {"status": "off", "response": "Lights are OFF now."}
    return {"status": "off", "response": "Lights are already OFF."}

# Create a state graph for light control
graph = StateGraph(LightControlState)

# Add light control actions as nodes
graph.add_node("light_status", light_status)
graph.add_node("turn_on_lights", turn_on_lights)
graph.add_node("turn_off_lights", turn_off_lights)

# Set initial entry point
graph.set_entry_point("light_status")

# Compile the graph
light_control_pipeline = graph.compile()

# Initialize global state once
light_state = LightControlState()

def control_lights(intent: str, command: str) -> str:
    global light_state

    print(f"💡 Light control command: {command}, intent: {intent}")

    if intent == "turn_on_lights":
        result = turn_on_lights(light_state)
        light_state = LightControlState(**result)

    elif intent == "turn_off_lights":
        result = turn_off_lights(light_state)
        light_state = LightControlState(**result)

    else:
        return "❌ Invalid command. Try: 'Turn on the lights' or 'Turn off the lights'."

    response = light_state.response
    print(f"💡 Light control response: {response}")
    return response
