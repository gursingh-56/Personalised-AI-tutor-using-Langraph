import os
import json
import re
from dotenv import load_dotenv
from pathlib import Path
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from quiz_user_node import quiz_user_node, review_agent_node

# ──────────────────────────────────────
# Setup
# ──────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1, google_api_key=GOOGLE_API_KEY)
chat_history = []

# ──────────────────────────────────────
# Helpers
# ──────────────────────────────────────
def build_system_prompt(state):
    preferences = state.get("preferences", {})
    topic = state.get("current_topic", "Unknown topic")
    level = state.get("current_level", "Unknown level")
    name = state.get("name", "User")

    system_prompt = f"""
You are a personalized AI tutor for {name}.
Their learning preferences are: {json.dumps(preferences, indent=2)}.

Topic to teach: {topic}
Current level: {level}

Adapt your style to be clear, brief, and actionable. Don't explain everything at once. Teach one core idea at a time, using small code examples or analogies. Wait for the student to ask for more or quiz them after each chunk. Avoid long answers or full lessons unless explicitly asked.
"""
    return {"role": "system", "content": system_prompt.strip()}

def convert_to_langchain_messages(history):
    converted = []
    for msg in history:
        if msg["role"] == "user":
            converted.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            converted.append(AIMessage(content=msg["content"]))
        elif msg["role"] == "system":
            converted.append(SystemMessage(content=msg["content"]))
    return converted

# ──────────────────────────────────────
# Nodes
# ──────────────────────────────────────
def get_info(state: dict) -> dict:
    name = input("Enter your Name: ").strip()
    state["name"] = name
    profile_path = Path("profiles") / f"{name}.json"
    os.makedirs("profiles", exist_ok=True)

    if profile_path.exists():
        with open(profile_path, "r") as f:
            print(f"\n✅ Loaded existing learning profile for {name}.")
            state["preferences"] = json.load(f)
            state["first_time"] = False
            return state

    # First-time user – ask learning preference questions
    question_prompt = """
    Ask me 15 questions about my learning preferences covering: information processing style, preferred content formats, 
    optimal study environment, motivation factors, memory techniques that work, attention span, and feedback preferences.
    Then create a JSON profile with specific recommendations for how AI should adapt to my style in future sessions.
    Do not answer them — only return questions as plain text, no bullet points or explanations.
    """
    question_response = llm.invoke([HumanMessage(content=question_prompt)])
    raw_questions = question_response.content.strip().split("\n")

    questions = []
    for q in raw_questions:
        q = q.strip()
        if not q:
            continue
        q = re.sub(r"^\s*\d+[\.\)]\s*", "", q)
        questions.append(q)
    if len(questions) < 10:
        print("❌ Not enough questions received.")
        exit(1)

    answers = {}
    for idx, q in enumerate(questions, 1):
        print(f"Q{idx}. {q}")
        answers[q] = input("➤ ")

    # Analyze answers
    analysis_prompt = f"""
    Based on the following user responses, give a JSON-formatted analysis of their learning preferences
    and how to teach them best (include methods, formats, environment, motivation style, etc.).
    Return ONLY a valid JSON object. No text before or after.

    Responses:
    {json.dumps(answers, indent=2)}
    """
    analysis_response = llm.invoke([HumanMessage(content=analysis_prompt)])
    try:
        raw = analysis_response.content.strip("```json").strip("```").strip()
        profile_json = json.loads(raw)
    except:
        print("❌ Could not parse learning profile. Exiting.")
        exit(1)

    with open(profile_path, "w") as f:
        json.dump(profile_json, f, indent=2)

    print(f"\n✅ Saved learning profile for {name} at {profile_path}")
    state["preferences"] = profile_json
    state["first_time"] = True
    return state

# ──────────────────────────────────────
# Stream Chat Updates
# ──────────────────────────────────────
def stream_graph_updates(user_input: str):
    global chat_history

    if user_input.startswith("!"):
        command = user_input[1:].strip().lower()
        if command == "quiz":
            print("\n📚 Launching quiz...\n")
            quiz_user_node(final_state)
            return
        elif command == "review":
            print("\n📘 Launching review...\n")
            review_agent_node(final_state)
            return
        elif command == "help":
            print("Commands: !quiz, !review, !help, q")
            return
        else:
            print("❌ Unknown command.")
            return

    chat_history.append({"role": "user", "content": user_input})
    lc_messages = convert_to_langchain_messages(chat_history)

    response = llm.invoke(lc_messages)
    print("Assistant:", response.content)

    chat_history.append({"role": "assistant", "content": response.content})


# ──────────────────────────────────────
# Graph Setup
# ──────────────────────────────────────
builder = StateGraph(dict)
builder.set_entry_point("get_info")
builder.add_node("get_info", RunnableLambda(get_info))
builder.add_edge("get_info", END)
graph = builder.compile()

# ──────────────────────────────────────
# INIT RUN
# ──────────────────────────────────────
final_state = graph.invoke({})

# Only ask topic & run quiz+review if new
if final_state.get("first_time"):
    final_state["current_topic"] = input("📘 What topic would you like to learn? ")
    final_state["current_level"] = input("📗 Your level (Beginner, Intermediate, Advanced): ")

    print("\n📚 Launching quiz...\n")
    final_state = quiz_user_node(final_state)

    print("\n📘 Reviewing quiz mistakes...\n")
    final_state = review_agent_node(final_state)

    print("\n✅ Quiz and review complete.")

# ──────────────────────────────────────
# System Prompt + Chat History
# ──────────────────────────────────────
system_msg = build_system_prompt(final_state)
chat_history = [system_msg]

chat_path = Path("chats") / f"{final_state['name']}.json"
if chat_path.exists():
    with open(chat_path, "r") as f:
        old = json.load(f)
        old = [msg for msg in old if msg["role"] != "system"]
        chat_history.extend(old)
        print(f"💬 Chat history loaded from {chat_path}")
else:
    os.makedirs("chats", exist_ok=True)

# ──────────────────────────────────────
# Chat Loop
# ──────────────────────────────────────
print("\nWelcome to the Interactive Session.")
print("You can now chat with your AI tutor or use commands (!quiz, !review, q to quit).")

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            with open(chat_path, "w") as f:
                json.dump([m for m in chat_history if m["role"] != "system"], f, indent=2)
            print(f"💾 Chat saved to {chat_path}")
            break
        stream_graph_updates(user_input)
    except:
        break
