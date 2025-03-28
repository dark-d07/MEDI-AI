import numpy as np
import json
import random
import sympy as sp
import re
import datetime
import os

# Load training data
if os.path.exists("bot.json"):
    with open("bot.json", "r") as file:
        data = json.load(file)
else:
    data = {"responses": {}, "medical_questions": {}, "jokes": [], "fun_facts": [], "medical_facts": []}

responses = data.get("responses", {})
medical_questions = data.get("medical_questions", {})
jokes = data.get("jokes", [])
fun_facts = data.get("fun_facts", [])
medical_facts = data.get("medical_facts", [])

# Merge all question categories
all_responses = {**responses, **medical_questions}

# Simple text similarity using NumPy
def text_similarity(user_input):
    best_match = None
    best_score = 0
    
    for question, answer in all_responses.items():
        words_input = set(user_input.lower().split())
        words_question = set(question.lower().split())
        score = len(words_input & words_question) / len(words_question)  # Intersection over total words
        
        if score > best_score:
            best_score = score
            best_match = answer
    
    return best_match if best_score > 0 else "I'm sorry, I don't understand that."

# Detect and solve math expressions
def detect_and_solve_math(user_input):
    user_input_cleaned = re.sub(r"(what is|calculate|find|solve|compute|answer)", "", user_input, flags=re.IGNORECASE).strip()
    user_input_cleaned = user_input_cleaned.replace("x", "*")

    if re.search(r"[\+\-\*/\^]", user_input_cleaned):
        try:
            result = sp.sympify(user_input_cleaned)
            return f"The answer is: {result}"
        except:
            return "Sorry, I couldn't solve that math problem."

    return None

# Get current time
def get_time():
    return "The current time is: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Main chatbot function
def get_response(user_input):
    user_input = user_input.strip().lower()

    # Check for math problems
    math_response = detect_and_solve_math(user_input)
    if math_response:
        return math_response

    # Process text queries
    response = text_similarity(user_input)
    
    # Special responses
    special_responses = {
        "get_joke": random.choice(jokes) if jokes else "No jokes available.",
        "get_fact": random.choice(fun_facts) if fun_facts else "No fun facts available.",
        "get_medical_fact": random.choice(medical_facts) if medical_facts else "No medical facts available.",
        "get_time": get_time()
    }
    
    return special_responses.get(response, response)

# Test chatbot in CLI
if __name__ == "__main__":
    print("ChatBot: Hello! Type 'quit' to exit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            print("ChatBot: Goodbye!")
            break
        print(f"ChatBot: {get_response(user_input)}")
