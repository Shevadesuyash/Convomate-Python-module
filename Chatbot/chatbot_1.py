# responses generated from given option
import language_tool_python
import random
import time


# Initialize grammar checker
tool = language_tool_python.LanguageTool('en-US')

# Base questions
starter_questions = [
    "Hey! How was your day?",
    "What did you eat today?",
    "What are your hobbies?",
    "Do you enjoy music or movies?",
    "What's your dream job?",
    "Have you traveled anywhere recently?"
]

# Follow-up questions mapped to topics
follow_ups = {
    "food": [
        "That sounds tasty! Do you enjoy cooking too?",
        "What’s your favorite dish?",
        "Do you prefer home-cooked meals or takeout?"
    ],
    "music": [
        "Nice! What genre do you listen to the most?",
        "Who's your favorite singer or band?",
        "Do you play any musical instruments?"
    ],
    "movies": [
        "Great! What kind of movies do you enjoy?",
        "Have you watched anything recently?",
        "Do you prefer watching at home or in the theater?"
    ],
    "travel": [
        "Awesome! Where did you go?",
        "Do you like traveling to mountains or beaches?",
        "Which place do you want to visit next?"
    ],
    "job": [
        "Cool! Why does that job interest you?",
        "Do you need any special skills for that?",
        "Are you studying for it currently?"
    ],
    "day": [
        "Sounds like a full day! What was the best part?",
        "Did anything surprising happen today?",
        "Do you usually have days like this?"
    ],
    "hobbies": [
        "That's interesting! How did you get into that?",
        "Do you do it daily or occasionally?",
        "Have you met others who enjoy the same hobby?"
    ]
}

# Friendly compliments
compliments = [
    "You're improving so well!",
    "Nice! That was much better.",
    "Good effort! You're doing awesome 😊",
    "Great! Just a little tweak needed.",
    "You're getting better with each sentence!"
]

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load model once at the beginning
tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")

def correct_grammar(text):
    input_text = "gec: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text


def detect_topic(text):
    text = text.lower()
    if any(word in text for word in ["eat", "food", "lunch", "dinner", "breakfast", "cook"]):
        return "food"
    elif any(word in text for word in ["music", "song", "singer", "band", "instrument"]):
        return "music"
    elif any(word in text for word in ["movie", "film", "series", "watch"]):
        return "movies"
    elif any(word in text for word in ["travel", "trip", "vacation", "journey"]):
        return "travel"
    elif any(word in text for word in ["job", "career", "work", "profession"]):
        return "job"
    elif any(word in text for word in ["today", "morning", "afternoon", "evening"]):
        return "day"
    elif any(word in text for word in ["hobby", "hobbies", "play", "draw", "paint", "read", "game"]):
        return "hobbies"
    else:
        return None

def chat():
    print("🤖: Hi there! I'm your offline English buddy. Let's practice English together!")
    time.sleep(1)

    question = random.choice(starter_questions)

    while True:
        print(f"\n🤖: {question}")
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("🤖: It was lovely chatting with you! Keep practicing 😊")
            break

        corrected = correct_grammar(user_input)

        if corrected.strip().lower() != user_input.strip().lower():
            print(f"🤖: Here's a better way to say it: \"{corrected}\"")
            print(f"🤖: {random.choice(compliments)}")
        else:
            print(f"🤖: That was perfect! 🎉")
            print(f"🤖: {random.choice(compliments)}")

        # Follow-up based on detected topic
        topic = detect_topic(user_input)
        if topic and topic in follow_ups:
            question = random.choice(follow_ups[topic])
        else:
            question = random.choice(starter_questions)

        time.sleep(1)

# Run it!
chat()
