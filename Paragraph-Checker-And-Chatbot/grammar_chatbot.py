from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import random
from typing import Dict, List

class GrammarChatbot:
    def __init__(self):
        """Initialize models for fluent corrections"""
        # Initialize models
        self.grammar_model = pipeline(
            "text2text-generation",
            model="vennify/t5-base-grammar-correction",
            device=0 if torch.cuda.is_available() else -1
        )

        # Fluent paraphrasing model
        self.paraphrase_tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
        self.paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

        # Enhanced conversation components
        self.compliments = [
            "Great job! Your English is improving!",
            "Nice improvement! Keep it up!",
            "You're doing well with your practice!",
            "Good effort! I can see you're trying hard!",
            "Excellent try! You're getting better!",
            "Well done! That was much better!",
            "Impressive! Your sentence structure is improving!"
        ]

        # Organized question bank by categories
        self.question_categories = {
            "daily_life": [
                "What did you do this morning?",
                "How do you usually spend your evenings?",
                "What's your morning routine like?",
                "Do you have any plans for this weekend?",
                "What time do you usually wake up?"
            ],
            "hobbies": [
                "What hobbies do you enjoy in your free time?",
                "Have you picked up any new hobbies recently?",
                "Do you prefer indoor or outdoor activities?",
                "What's something you've always wanted to try?",
                "Do you play any musical instruments?"
            ],
            "food": [
                "What's your favorite comfort food?",
                "Do you enjoy cooking? What's your specialty?",
                "What's the most unusual food you've ever tried?",
                "Do you prefer sweet or savory snacks?",
                "What's your go-to breakfast?"
            ],
            "travel": [
                "If you could visit any country, where would you go?",
                "What's the most beautiful place you've ever seen?",
                "Do you prefer beach vacations or city trips?",
                "What's your favorite travel memory?",
                "What's the next place you'd like to visit?"
            ],
            "technology": [
                "How do you use technology in your daily life?",
                "What's your opinion about social media?",
                "Do you think AI will change our lives significantly?",
                "What tech gadget couldn't you live without?",
                "How has technology changed your work/studies?"
            ],
            "future": [
                "Where do you see yourself in five years?",
                "What's one skill you'd like to learn?",
                "Do you have any big goals for this year?",
                "What would your perfect day look like?",
                "What's something you want to achieve?"
            ]
        }

        self.current_question = None
        self.current_category = None
        self.conversation_history = []

    def correct_grammar(self, text: str) -> str:
        """Basic grammar correction"""
        result = self.grammar_model(
            text,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
        return result[0]['generated_text']

    def fluent_correction(self, text: str) -> str:
        """Create fluent, natural rewrites"""
        input_ids = self.paraphrase_tokenizer(
            f"paraphrase: {text}",
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).input_ids

        outputs = self.paraphrase_model.generate(
            input_ids,
            temperature=0.7,
            max_length=256,
            num_beams=5,
            early_stopping=True
        )

        return self.paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _get_next_question(self) -> str:
        """Select next question based on conversation context"""
        # If we're continuing a category, use follow-up questions
        if self.current_category and random.random() < 0.6:  # 60% chance to stay on topic
            return random.choice(self.question_categories[self.current_category])

        # Otherwise select a new random category
        self.current_category = random.choice(list(self.question_categories.keys()))
        return random.choice(self.question_categories[self.current_category])

    def generate_response(self, user_input: str) -> Dict:
        """Generate a conversational response"""
        # Store conversation history
        self.conversation_history.append(("user", user_input))

        # Correct the input
        corrected = self.fluent_correction(user_input)
        is_corrected = corrected.lower() != user_input.lower()

        # Generate response
        response = {
            "original_text": user_input,
            "corrected_text": corrected,
            "is_corrected": is_corrected,
            "compliment": random.choice(self.compliments) if is_corrected else "",
            "next_question": self._get_next_question(),
            "end_conversation": False
        }

        # Update state
        self.current_question = response["next_question"]
        self.conversation_history.append(("bot", response["next_question"]))

        return response

    def start_conversation(self) -> Dict:
        """Initialize a new conversation"""
        self.conversation_history = []
        self.current_category = random.choice(list(self.question_categories.keys()))
        self.current_question = random.choice(self.question_categories[self.current_category])

        return {
            "message": "Hello! I'm your English practice partner. Let's chat!",
            "next_question": self.current_question,
            "end_conversation": False
        }

    def get_conversation_history(self) -> List[tuple]:
        """Get the complete conversation history"""
        return self.conversation_history