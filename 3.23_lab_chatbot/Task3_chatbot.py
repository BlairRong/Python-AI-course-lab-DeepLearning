
# A simple Python chatbot that greets user by name with format instractions.

import random

def get_name():
    """Ask the user for their name and greet them."""
    name = input("Hello! What's your name? ").strip()
    if name:
        print(f"Nice to meet you, {name}! I'm your chatbot. Type 'bye' to exit.")
    else:
        print("Nice to meet you! I'm your chatbot. Type 'bye' to exit.")
    return name

def get_response(user_input):
    """Return a response based on keywords in the input."""
    msg = user_input.lower()

    # Greetings
    if any(word in msg for word in ["hello", "hi", "hey"]):
        return random.choice(["Hello!", "Hi there!", "Hey!"])

    # Asking about the chatbot
    elif "your name" in msg or "who are you" in msg:
        return "I'm a simple Python chatbot. You can call me PyBot."

    # How are you
    elif "how are you" in msg:
        return "I'm doing great, thanks! How about you?"

    # Weather (mock)
    elif "weather" in msg:
        return "I don't have live weather data, but I hope it's nice!"

    # Goodbye
    elif any(word in msg for word in ["bye", "goodbye", "see you"]):
        return None  # Signal to exit

    # Default fallback
    else:
        return "Interesting! Tell me more."

def main():
    """Run the chatbot."""
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
    main()



#observation the code structure with Task 2
"""
The two final chatbot versions are structurally very similar, but there are a few key differences:

1. **Function naming**  
- The previous version used a function called `chat()` to run the main loop.  
- The latest version uses a function called `main()`, which is a more conventional name for the entry point.

2. **Main loop placement**  
- In the previous version, `chat()` contained the loop, and `if __name__ == "__main__": chat()` invoked it.  
- In the latest version, the loop is inside `main()` and called directly; there is no separate `chat()` function.

3. **Line count and compactness**  
- The latest version is slightly more condensed (under 50 lines) by merging some print statements 
and removing a few blank lines, while still keeping comments and readability.

4. **Greeting handling**  
- Both ask for a name, but the latest version prints the greeting in a single line without an extra blank line, 
making it slightly tighter.

Overall, the structure remains functional and modular, but the latest version adopts a more compact style 
with a single entry function named `main()`, which is typical for small Python scripts.
"""