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
        self.starter_questions = [
            {"text": "Hey there! How was your day today?", "topic": "day"},
            {"text": "What did you eat for your last meal, and did you enjoy it?", "topic": "food"},
            {"text": "What are some of your favorite hobbies or activities you enjoy doing?", "topic": "hobbies"},
            {"text": "Do you enjoy listening to music or watching movies more? Why?", "topic": "general_entertainment"},
            {"text": "If you could have any job in the world, what would it be and why?", "topic": "job"},
            {"text": "Have you traveled anywhere interesting recently, or where would you absolutely love to go next?", "topic": "travel"},
            {"text": "What's one thing you're looking forward to this week?", "topic": "future_plans"},
            {"text": "Do you prefer reading books or articles online?", "topic": "reading"},
            {"text": "What's a skill you'd like to learn in the future?", "topic": "learning_skills"},
            {"text": "Tell me about a favorite memory you have.", "topic": "memories"}
        ]

        self.follow_ups = {
            "food": [
                {"text": "That sounds delicious! Do you enjoy cooking, or do you prefer eating out?", "topic": "food"},
                {"text": "What's your absolute favorite dish, and what makes it so special?", "topic": "food"},
                {"text": "Is there any food you absolutely dislike or refuse to eat?", "topic": "food"},
                {"text": "Do you have a go-to recipe you love to make?", "topic": "food"},
                {"text": "What's the most unusual food you've ever tried?", "topic": "food"}
            ],
            "music": [
                {"text": "Nice! What genre of music do you listen to the most, and why are you drawn to it?", "topic": "music"},
                {"text": "Who's your favorite artist or band, and what's your favorite song by them?", "topic": "music"},
                {"text": "Do you play any musical instruments, or wish you could?", "topic": "music"},
                {"text": "What kind of music do you listen to when you need to relax?", "topic": "music"},
                {"text": "Is there a concert or music festival you'd love to attend?", "topic": "music"}
            ],
            "movies": [
                {"text": "Great! What kind of movies do you enjoy? (e.g., action, comedy, sci-fi, drama) Can you tell me about a recent one you liked?", "topic": "movies"},
                {"text": "Have you watched any good movies or series recently that you'd highly recommend? What made them good?", "topic": "movies"},
                {"text": "Do you prefer watching movies at home, or do you enjoy the cinema experience more?", "topic": "movies"},
                {"text": "Who are some of your favorite actors or directors?", "topic": "movies"},
                {"text": "What's a classic movie you think everyone should watch?", "topic": "movies"}
            ],
            "travel": [
                {"text": "Awesome! Where did you go, or where do you dream of visiting? What's appealing about that place?", "topic": "travel"},
                {"text": "Do you prefer traveling to mountains, beaches, or bustling cities, and why?", "topic": "travel"},
                {"text": "What's the most memorable trip you've ever taken, and what made it so memorable?", "topic": "travel"},
                {"text": "Do you enjoy planning your trips in detail, or do you prefer to be spontaneous?", "topic": "travel"},
                {"text": "What's one thing you always bring with you when you travel?", "topic": "travel"}
            ],
            "job": [
                {"text": "Cool! What specifically makes that job appealing to you?", "topic": "job"},
                {"text": "What kind of skills or education do you think are most important for that profession?", "topic": "job"},
                {"text": "Are you currently working towards your dream job, or planning to?", "topic": "job"},
                {"text": "What's the biggest challenge you anticipate in that job?", "topic": "job"},
                {"text": "If you weren't doing that, what would be your second choice of career?", "topic": "job"}
            ],
            "day": [
                {"text": "Sounds like a full day! What was the best part of your day, or something that made you smile?", "topic": "day"},
                {"text": "Did anything interesting or surprising happen today?", "topic": "day"},
                {"text": "How do you usually unwind after a long day?", "topic": "day"},
                {"text": "Is there anything you wish you had more time for in your day?", "topic": "day"}
            ],
            "hobbies": [
                {"text": "That's interesting! How did you get into that hobby, and what do you enjoy most about it?", "topic": "hobbies"},
                {"text": "How often do you get to engage in your hobbies, and with whom?", "topic": "hobbies"},
                {"text": "Have you met anyone else who shares your hobby, or found a community around it?", "topic": "hobbies"},
                {"text": "What's the most challenging aspect of your hobby?", "topic": "hobbies"},
                {"text": "Is there a new hobby you're thinking of trying?", "topic": "hobbies"}
            ],
            "general_entertainment": [ # New broader category for movies/music initial question
                {"text": "If you picked music, what's your go-to genre? If movies, what's your favorite film genre? Tell me more!", "topic": "music_or_movies"},
                {"text": "Do you prefer listening to music while working, or when relaxing?", "topic": "music"},
                {"text": "Are you more into watching TV series or standalone movies?", "topic": "movies"},
                {"text": "What kind of entertainment helps you de-stress?", "topic": "general_entertainment"}
            ],
            "future_plans": [
                {"text": "That sounds exciting! What steps are you taking to achieve that?", "topic": "future_plans"},
                {"text": "Is there a specific goal you're working towards?", "topic": "future_plans"},
                {"text": "How do you stay motivated to work on your plans?", "topic": "future_plans"}
            ],
            "reading": [
                {"text": "What kind of books or articles do you enjoy reading the most?", "topic": "reading"},
                {"text": "Do you have a favorite author or genre?", "topic": "reading"},
                {"text": "What are you currently reading?", "topic": "reading"}
            ],
            "learning_skills": [
                {"text": "That's a great goal! Why does that skill interest you?", "topic": "learning_skills"},
                {"text": "How do you plan to go about learning it?", "topic": "learning_skills"},
                {"text": "Do you think it will be challenging or easy to learn?", "topic": "learning_skills"}
            ],
            "memories": [
                {"text": "That sounds lovely! Can you tell me more details about that memory?", "topic": "memories"},
                {"text": "Who were you with, and what made it so special?", "topic": "memories"},
                {"text": "Do you often think about this memory?", "topic": "memories"}
            ]
        }

        # Friendly compliments
        self.compliments = [
            "You're making great progress! Keep it up!",
            "Excellent effort! That was much clearer.",
            "Fantastic! You're really improving your sentences.",
            "Very good! Just a minor adjustment there.",
            "You're becoming more fluent with every practice!",
            "Your sentences are getting stronger!",
            "That was a well-constructed thought!",
            "Impressive clarity in your writing!",
            "You're mastering this step by step!"
        ]

        # Initialize conversation state
        self.current_question = random.choice(self.starter_questions)["text"]
        self.conversation_history = [] # To store (user_input, bot_response_object) tuples

    def correct_grammar(self, text):
        """Correct grammar of the input text"""
        if not text.strip():
            return text # Return original if empty
        input_text = "gec: " + text
        try:
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = self.model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error during grammar correction: {e}")
            return text # Fallback to original text on error

    def detect_topic(self, text):
        """
        Detect the topic of the user's input.
        More comprehensive keyword lists could be loaded from a config or database.
        """
        text = text.lower()
        if any(word in text for word in ["eat", "food", "lunch", "dinner", "breakfast", "cook", "meal", "dish", "restaurant", "recipe"]):
            return "food"
        elif any(word in text for word in ["music", "song", "singer", "band", "instrument", "genre", "artist", "concert", "headphones"]):
            return "music"
        elif any(word in text for word in ["movie", "film", "series", "watch", "cinema", "show", "actor", "director", "tv"]):
            return "movies"
        elif any(word in text for word in ["travel", "trip", "vacation", "journey", "visit", "destination", "explore", "holiday", "airport", "beach", "mountain", "city"]):
            return "travel"
        elif any(word in text for word in ["job", "career", "work", "profession", "study", "student", "occupation", "company", "boss", "salary", "degree"]):
            return "job"
        elif any(word in text for word in ["today", "morning", "afternoon", "evening", "yesterday", "day", "week", "weekend"]):
            return "day"
        elif any(word in text for word in ["hobby", "hobbies", "play", "draw", "paint", "read", "game", "activity", "interest", "leisure", "craft", "sport"]):
            return "hobbies"
        elif any(word in text for word in ["entertainment", "fun", "leisure", "relax", "enjoy"]):
            return "general_entertainment"
        elif any(word in text for word in ["future", "plans", "looking forward", "goal", "next week", "tomorrow"]):
            return "future_plans"
        elif any(word in text for word in ["read", "book", "article", "story", "novel", "magazine", "library"]):
            return "reading"
        elif any(word in text for word in ["learn", "skill", "study", "improve", "master", "develop"]):
            return "learning_skills"
        elif any(word in text for word in ["memory", "remember", "past", "experience", "childhood", "event"]):
            return "memories"
        return None

    def generate_response(self, user_input):
        """Generate a response to the user's input"""
        user_input_lower = user_input.lower().strip()

        # Check for exit commands
        if user_input_lower in ["exit", "quit", "bye", "goodbye", "end conversation"]:
            return {
                "original_text": user_input,
                "corrected_text": user_input,
                "is_corrected": False,
                "compliment": "",
                "next_question": "It was lovely chatting with you! Keep practicing and have a great day! 😊",
                "end_conversation": True
            }

        # Correct the user's grammar
        corrected = self.correct_grammar(user_input)

        # Prepare response
        response = {
            "original_text": user_input,
            "corrected_text": corrected,
            "is_corrected": corrected.strip().lower() != user_input_lower,
            "compliment": random.choice(self.compliments) if corrected.strip().lower() != user_input_lower else "Well done! Your grammar is perfect!",
            "end_conversation": False
        }

        # Determine next question based on topic
        topic = self.detect_topic(user_input)
        if topic and topic in self.follow_ups:
            next_q_options = self.follow_ups[topic]
            # Ensure we don't ask the same question repeatedly if possible
            if self.conversation_history:
                last_question_text = self.conversation_history[-1]['bot_response']['next_question']
                available_questions = [q for q in next_q_options if q["text"] != last_question_text]
                if available_questions:
                    response["next_question"] = random.choice(available_questions)["text"]
                else: # Fallback if all follow-ups have been asked
                    response["next_question"] = random.choice(self.starter_questions)["text"]
            else:
                response["next_question"] = random.choice(next_q_options)["text"]
        else:
            # If no specific topic detected, or no follow-ups for it, pick a random starter question
            response["next_question"] = random.choice(self.starter_questions)["text"]

        # Update current question (for the bot's internal state)
        self.current_question = response["next_question"]

        # Add to conversation history
        self.conversation_history.append({"user_input": user_input, "bot_response": response})

        return response

    def start_conversation(self):
        """Start a new conversation"""
        self.conversation_history = [] # Clear history on start
        self.current_question = random.choice(self.starter_questions)["text"]
        return {
            "response": "Hi there! I'm your English practice buddy. Let's chat!",
            "next_question": self.current_question,
            "end_conversation": False
        }