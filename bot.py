import json
import random
import sympy as sp
import re
import datetime
import pickle
import numpy as np
import os
import subprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load training data from JSON file
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

# Prepare corpus and responses
corpus = list(all_responses.keys())  # User inputs
response_values = list(all_responses.values())  # Bot responses

# Chat history file
CHAT_HISTORY_FILE = "chat_history.json"

def save_chat(user_input, bot_response):
    """Saves chat history to a JSON file."""
    history = []
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            history = json.load(file)
    
    history.append({"user": user_input, "bot": bot_response})
    
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)

# Check if model exists
MODEL_FILE = "chatbot_model.pkl"

def train_and_save_model():
    """Trains the model and saves it to a file."""
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(corpus).toarray()
    
    with open(MODEL_FILE, "wb") as model_file:
        pickle.dump((vectorizer, vectors, response_values), model_file)

def load_model():
    """Loads the trained model if it exists."""
    try:
        with open(MODEL_FILE, "rb") as model_file:
            return pickle.load(model_file)
    except FileNotFoundError:
        return None

# Load or train model
model_data = load_model()
if model_data:
    vectorizer, vectors, response_values = model_data
else:
    train_and_save_model()
    model_data = load_model()
    vectorizer, vectors, response_values = model_data

# Function to detect and solve math expressions
def detect_and_solve_math(user_input):
    user_input_cleaned = re.sub(r"(what is|calculate|do|find|solve|compute|work out|answer)", "", user_input, flags=re.IGNORECASE).strip()
    user_input_cleaned = user_input_cleaned.replace("x", "*")
    
    if re.search(r"[\+\-\*/\^]", user_input_cleaned):
        try:
            result = sp.sympify(user_input_cleaned)
            return f"The answer is: {result}"
        except (sp.SympifyError, TypeError):
            return "Sorry, I couldn't solve that math problem."
    
    return None

# Function to get current system time
def get_time():
    return "The current time is: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Function to check health parameters using Raspberry Pi
def check_health():
    """Runs health_check.py and returns live JSON output."""
    try:
        process = subprocess.Popen(["python3", "health_check.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        process.wait()  # Ensure process completes before reading output
        stdout, stderr = process.communicate()  # Read full output

        try:
            data = json.loads(stdout.strip())  # Ensure clean JSON parsing
        except json.JSONDecodeError:
            return "Error: Invalid JSON format received from health_check.py"

        if not data:
            return "Error: No valid health data received."

        return (f"📊 Health Check Data:\n"
                f"🌡️ Ambient Temp: {data['ambient_temp_C']}°C ({data['ambient_temp_F']}°F)\n"
                f"🔥 Forehead Temp: {data['forehead_temp_C']}°C ({data['forehead_temp_F']}°F)")
    except Exception as e:
        return f"Error checking health: {str(e)}"

# Function to get chatbot response
def chatbot_response(user_input):
    user_input = user_input.strip().lower()  # Normalize input

    # Check if the input is a math problem
    math_response = detect_and_solve_math(user_input)
    if math_response:
        save_chat(user_input, math_response)
        return math_response
    
    # Check for health-related queries
    health_keywords = ["check my health", "i have", "i feel", "fever", "heart rate", "temperature"]
    if any(keyword in user_input for keyword in health_keywords):
        health_response = check_health()
        save_chat(user_input, health_response)
        return health_response
    
    # Process text-based queries
    user_vector = vectorizer.transform([user_input]).toarray()
    similarities = cosine_similarity(user_vector, vectors)
    best_match_index = np.argmax(similarities)
    
    if similarities[0][best_match_index] == 0:
        bot_response = "I'm sorry, I don't understand that."
    else:
        bot_response = response_values[best_match_index]
    
    # Check for special responses
    special_responses = {
        "get_joke": random.choice(jokes),
        "get_fact": random.choice(fun_facts),
        "get_medical_fact": random.choice(medical_facts),
        "get_time": get_time()
    }
    bot_response = special_responses.get(bot_response, bot_response)

    save_chat(user_input, bot_response)
    return bot_response

# Run chatbot
if __name__ == "__main__":
    print("ChatBot: Hello! Type 'quit' to exit.")
    while True:
        user_input = input("You: ").strip()
        print(f"User: {user_input}")
        if user_input.lower() == "quit":
            print("ChatBot: Goodbye!")
            break
        print(f"ChatBot: {chatbot_response(user_input)}")
