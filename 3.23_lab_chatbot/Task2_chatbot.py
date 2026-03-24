# A simple terminal chatbot that greets user by name and responds to basic keywords.

import random

def get_name():
    """Ask the user for their name and greet them."""
    name = input("Hello! What's your name? ").strip()
    if name:
        print(f"Nice to meet you, {name}! I'm your friendly chatbot. Type 'bye' to exit.\n")
    else:
        print("Nice to meet you! I'm your friendly chatbot. Type 'bye' to exit.\n")
    return name

def get_response(user_input):
    """Return a response based on keywords in the user's input."""
    user_input = user_input.lower()

    # Greetings
    if any(word in user_input for word in ["hello", "hi", "hey"]):
        return random.choice(["Hello!", "Hi there!", "Hey! How can I help?"])

    # How are you
    elif "how are you" in user_input:
        return "I'm doing great, thanks! How about you?"

    # Name question
    elif "your name" in user_input or "who are you" in user_input:
        return "I'm a simple Python chatbot. You can call me PyBot."

    # Weather (mock)
    elif "weather" in user_input:
        return "I don't have live weather data, but I hope it's nice wherever you are!"

    # Goodbye
    elif any(word in user_input for word in ["bye", "goodbye", "see you"]):
        return None  # Signal to exit

    # Default fallback
    else:
        return "Interesting! Tell me more."

def chat():
    """Main loop for the chatbot."""
    name = get_name()

    while True:
        user_input = input("You: ")
        response = get_response(user_input)

        if response is None:
            print(f"PyBot: Goodbye, {name}! Have a great day.")
            break
        else:
            print(f"PyBot: {response}")

if __name__ == "__main__":
    chat()



#observation
"""
The updated chatbot now **greets the user by name** and uses it in the farewell, making the interaction feel more personal.  

However, the core conversation logic remains **unchanged**: it still relies on simple keyword matching. Many inputs still fall back to the default `"Interesting! Tell me more."`, and it cannot handle complex questions or maintain context.  

In short: **better personalisation, same conversational limitations**. 
"""
