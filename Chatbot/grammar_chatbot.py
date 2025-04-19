import random
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class GrammarChatbot:
    def __init__(self):
        # Initialize grammar correction model
        self.tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")

        # Conversation data
        # Base questions
        self.starter_questions = [
            "Hey! How was your day?",
            "What did you eat today?",
            "What are your hobbies?",
            "Do you enjoy music or movies?",
            "What's your dream job?",
            "Have you traveled anywhere recently?"
        ]

        # Follow-up questions mapped to topics
        self.follow_ups = {
            "food": [
                "That sounds tasty! Do you enjoy cooking too?",
                "What's your favorite dish?",
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
        self.compliments = [
            "You're improving so well!",
            "Nice! That was much better.",
            "Good effort! You're doing awesome 😊",
            "Great! Just a little tweak needed.",
            "You're getting better with each sentence!"
        ]

        self.current_question = random.choice(self.starter_questions)

    def correct_grammar(self, text):
        """Correct grammar of the input text"""
        input_text = "gec: " + text
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def detect_topic(self, text):
        """Detect the topic of the user's input"""
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
        return None

    def generate_response(self, user_input):
        """Generate a response to the user's input"""
        # Check for exit commands
        if user_input.lower() in ["exit", "quit", "bye"]:
            return {
                "response": "It was lovely chatting with you! Keep practicing 😊",
                "end_conversation": True
            }

        # Correct the user's grammar
        corrected = self.correct_grammar(user_input)

        # Prepare response
        response = {
            "original_text": user_input,
            "corrected_text": corrected,
            "is_corrected": corrected.strip().lower() != user_input.strip().lower(),
            "compliment": random.choice(self.compliments),
            "end_conversation": False
        }

        # Determine next question based on topic
        topic = self.detect_topic(user_input)
        if topic and topic in self.follow_ups:
            response["next_question"] = random.choice(self.follow_ups[topic])
        else:
            response["next_question"] = random.choice(self.starter_questions)

        # Update current question
        self.current_question = response["next_question"]

        return response

    def start_conversation(self):
        """Start a new conversation"""
        self.current_question = random.choice(self.starter_questions)
        return {
            "response": "Hi there! I'm your English practice buddy. Let's chat!",
            "next_question": self.current_question,
            "end_conversation": False
        }