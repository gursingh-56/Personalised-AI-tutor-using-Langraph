
import json
import os
import re
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1, google_api_key = GOOGLE_API_KEY)

def quiz_user_node(state):
    quiz_mode = state.get("quiz_mode", "generate")  # default to generation

    if quiz_mode == "generate":
        topic = state.get("current_topic", "general")
        level = state.get("current_level", "beginner")
        preferences = state.get("preferences", {})

        prompt = f"""
        Based on the user's learning preferences: {json.dumps(preferences)},
        generate a short quiz (10 questions) on the topic: {topic}.
        Difficulty level: {level}.
        Format JSON:
        {{
            "questions": [
                {{
                    "question": "...",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer": "B"
                }},
                ...
            ]
        }}
        """

        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip().strip("```json").strip("```").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)

        try:
            quiz = json.loads(raw)
        except json.JSONDecodeError:
            print("❌ Failed to parse Gemini's quiz response as JSON.")
            print("Response was:\n", response.content)
            raise
        state["last_quiz"] = quiz

        # Ask user for answers here
        user_answers = []
        option_labels = ["A", "B", "C", "D"]

        for i, q in enumerate(quiz["questions"]):
            print(f"\nQ{i+1}: {q['question']}")
            for j, opt in enumerate(q["options"]):
                print(f"{option_labels[j]}. {opt}")

            while True:
                ans = input("Your answer (A/B/C/D): ").strip().upper()
                if ans in option_labels[:len(q["options"])]:
                    break
                print("Invalid input. Please enter A, B, C, or D.")

            selected_option = q["options"][option_labels.index(ans)]
            user_answers.append(selected_option)

        state["user_answers"] = user_answers
        state["quiz_mode"] = "evaluate"  # switch mode

        # Call evaluation inline
        return quiz_user_node(state)

    elif quiz_mode == "evaluate":
        quiz = state.get("last_quiz")
        user_answers = state.get("user_answers")
        level = state.get("current_level", "beginner")

        if not quiz or not user_answers or len(quiz["questions"]) != len(user_answers):
            raise ValueError("Mismatch in questions and answers or missing data.")

        results = []
        correct_count = 0

        for i, q in enumerate(quiz["questions"]):
            user_answer = user_answers[i]
            correct_answer = q["correct_answer"]

            result = {
                "question": q["question"],
                "user_answer": user_answer,
                "correct_answer": correct_answer,
                "is_correct": user_answer == correct_answer
            }

            

            if result["is_correct"]:
                correct_count += 1
            else:
                incorrect = []
                incorrect.append(q)
                state["review_needed"] = incorrect 
            results.append(result)

        score = correct_count / len(quiz["questions"]) * 100

        print(f"\nYour score: {score:.2f}%")

        state.setdefault("profile", {})["level"] = level
        state["last_results"] = results
        state.setdefault("quiz_history", []).append({
            "score": score,
            "level": level,
            "results": results,
        })
        return state

    else:
        raise ValueError("Invalid quiz_mode. Use 'generate' or 'evaluate'.")
    
def review_agent_node(state):
    incorrect_questions = state.get("review_needed", [])
    if not incorrect_questions:
        print("Nothing to review.")
        return state

    review_prompt = f"""
    The user got these questions wrong:
    {json.dumps(incorrect_questions, indent=2)}

    For each one, explain the correct answer simply and give a short explanation to help them understand.
    """
    response = llm.invoke([HumanMessage(content=review_prompt)])
    print("\nReview:\n", response.content)
    return state