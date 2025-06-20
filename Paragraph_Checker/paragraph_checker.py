# paragraph_checker.py
import language_tool_python
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Optional

class ParagraphCorrector:
    def __init__(self):
        """Initialize all correction models"""
        print("Initializing correction models...")

        # Initialize LanguageTool for grammar/spelling with larger timeout
        self.grammar_tool = language_tool_python.LanguageTool('en-US', config={'maxTextLength': 100000})

        # Initialize neural grammar correction model
        self.grammar_model = pipeline(
            "text2text-generation",
            model="vennify/t5-base-grammar-correction",
            device=0 if torch.cuda.is_available() else -1
        )

        # Initialize paraphrase model for tense/style
        self.paraphrase_tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
        self.paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

        print("All models loaded successfully!")

    def grammar_correction(self, text: str) -> str:
        """Correct grammar and spelling using LanguageTool"""
        if len(text) > 20000:  # Split very large texts
            return self._process_large_text(text, self.grammar_correction)
        matches = self.grammar_tool.check(text)
        return language_tool_python.utils.correct(text, matches)

    def neural_grammar_correction(self, text: str) -> str:
        """Correct grammar using neural model"""
        if len(text) > 1000:  # Split larger texts for neural model
            return self._process_large_text(text, self.neural_grammar_correction)
        result = self.grammar_model(
            text,
            max_length=512,  # Increased max length
            num_beams=5,
            repetition_penalty=1.0,
            early_stopping=True
        )
        return result[0]['generated_text']

    def paraphrase_text(self, text: str, temperature: float = 0.7) -> str:
        """Paraphrase text with controlled randomness for tense/style"""
        if len(text) > 1000:  # Split larger texts
            return self._process_large_text(text, lambda t: self.paraphrase_text(t, temperature))

        input_ids = self.paraphrase_tokenizer(
            f"paraphrase: {text}",
            return_tensors="pt",
            padding="longest",
            max_length=512,  # Increased max length
            truncation=True
        ).input_ids

        outputs = self.paraphrase_model.generate(
            input_ids,
            temperature=temperature,
            repetition_penalty=1.0,
            num_return_sequences=1,
            do_sample=True,
            max_length=512  # Increased max length
        )

        return self.paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _process_large_text(self, text: str, processor: callable, chunk_size: int = 500) -> str:
        """Process large text in chunks"""
        sentences = text.split('. ')
        result = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    result.append(processor(current_chunk))
                current_chunk = sentence + ". "

        if current_chunk:
            result.append(processor(current_chunk))

        return ' '.join(result)

    def correct_paragraph(self, text: str, aggressive: bool = False) -> str:
        """
        Complete paragraph correction pipeline
        :param text: Input text to correct
        :param aggressive: Whether to use more aggressive neural corrections
        :return: Corrected text
        """
        if not text.strip():
            return text

        # Step 1: Basic grammar/spelling correction
        corrected = self.grammar_correction(text)

        # Step 2: Neural grammar correction (more comprehensive)
        if aggressive:
            corrected = self.neural_grammar_correction(corrected)
            # Re-check with LanguageTool
            corrected = self.grammar_correction(corrected)

        # Step 3: Tense/style correction (carefully applied)
        paraphrased = self.paraphrase_text(corrected)

        # Final grammar check
        final_output = self.grammar_correction(paraphrased)

        return final_output

# Global instance for Flask app
corrector = ParagraphCorrector()

def initialize_models():
    """Initialize function for Flask app"""
    global corrector
    corrector = ParagraphCorrector()

def correct_paragraph(text: str, aggressive: bool = False) -> str:
    """Wrapper function for Flask app"""
    return corrector.correct_paragraph(text, aggressive)