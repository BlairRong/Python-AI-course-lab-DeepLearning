# simple_chatbot.py

import random

def get_response(user_input):
    user_input = user_input.lower()

    # Greetings
    if any(word in user_input for word in ["hello", "hi", "hey"]):
        return random.choice(["Hello!", "Hi there!", "Hey! How can I help?"])

    # How are you
    elif "how are you" in user_input:
        return "I'm doing great, thanks! How about you?"

    # Asking about name
    elif "your name" in user_input or "who are you" in user_input:
        return "I'm a simple Python chatbot. You can call me PyBot."

    # Weather
    elif "weather" in user_input:
        return "I don't have access to live data, but I hope it's nice wherever you are!"

    # Goodbye
    elif any(word in user_input for word in ["bye", "goodbye", "see you"]):
        return "Goodbye! Have a great day."

    # Default / fallback
    else:
        return "Interesting! Tell me more."

def chat():
    print("Hello! I'm your simple Python chatbot. Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "exit", "quit"]:
            print("PyBot: Goodbye!")
            break
        response = get_response(user_input)
        print(f"PyBot: {response}")

if __name__ == "__main__":
    chat()



#run
#python simple_chatbot.py


#observation:
"""
**Does it work?**  
Yes, the chatbot runs and responds correctly to basic inputs like greetings, name queries, and farewells.

**Does it do exactly what you want?**  
Not exactly. It handles only a few predefined patterns; anything outside those patterns triggers the default fallback (`"Interesting! Tell me more."`).

**What is missing or wrong?**
- No real understanding of context (e.g., it doesn't remember I said "ok pybot" later).
- Cannot answer questions about live data (e.g., tempreture; weather in Gothenburg).
- Limited to simple keyword matching; fails on complex or nuanced questions.
- The fallback response is repetitive and unnatural.
- No ability to handle multi-turn conversations or ask clarifying questions.
"""