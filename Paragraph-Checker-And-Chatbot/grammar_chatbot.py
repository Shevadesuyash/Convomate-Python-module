from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import random

class GrammarChatbot:
    def __init__(self):
        """Initialize models for fluent corrections"""
        # Grammar correction model
        self.grammar_model = pipeline(
            "text2text-generation",
            model="vennify/t5-base-grammar-correction",
            device=0 if torch.cuda.is_available() else -1
        )

        # Fluent paraphrasing model
        self.paraphrase_tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
        self.paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

        # Conversation components
        self.compliments = [
            "Great job!",
            "Nice improvement!",
            "You're doing well!",
            "Good effort!",
            "Excellent try!"
        ]
        self.questions = [
            "What are you doing today?",
            "Have you read any good books lately?",
            "What's your favorite season?",
            "Do you have any pets?",
            "What's your favorite type of food?"
        ]
        self.current_question = None

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
            temperature=0.7,  # Medium creativity
            max_length=256,
            num_beams=5,
            early_stopping=True
        )

        return self.paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_response(self, user_input: str) -> dict:
        """Generate a conversational response"""
        # Correct and improve the input fluently
        corrected = self.fluent_correction(user_input)

        # Determine if correction was needed
        is_corrected = corrected.lower() != user_input.lower()

        # Prepare response
        response = {
            "original_text": user_input,
            "corrected_text": corrected,
            "is_corrected": is_corrected,
            "compliment": random.choice(self.compliments) if is_corrected else "",
            "next_question": self.current_question or random.choice(self.questions),
            "end_conversation": False
        }

        # Update current question
        self.current_question = random.choice(self.questions)

        return response

    def start_conversation(self) -> dict:
        """Initialize a new conversation"""
        self.current_question = random.choice(self.questions)
        return {
            "message": "Hello! Let's practice English. I'll help correct your sentences.",
            "next_question": self.current_question,
            "end_conversation": False
        }