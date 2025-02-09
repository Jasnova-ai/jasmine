import re
from chat import send_command_to_ai
from intent import classify_intent
from agents.agent_manager import execute_agent

def main():
    while True:
        command = input("\n🗣️ Enter command: ").strip()
        if command.lower() in ["exit", "quit"]:
            break

        # Classify intent (LLM chat or agent)
        intent = classify_intent(command)

        print(f"🤖 Intent: {intent}")

        if intent == "llm" or intent == "other":
            send_command_to_ai(command)
        else:
            execute_agent(intent, command)

def cleanup():
    """Gracefully stops all resources."""
    print("🛑 Cleaning up resources...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Exiting...")
        cleanup()
        exit(0)