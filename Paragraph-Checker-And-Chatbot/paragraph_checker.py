import language_tool_python
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class ParagraphCorrector:
    def __init__(self):
        """Initialize correction models with conservative settings"""
        # Grammar tool with increased timeout
        self.grammar_tool = language_tool_python.LanguageTool(
            'en-US',
            config={'maxTextLength': 100000}
        )

        # Conservative grammar correction model
        self.grammar_model = pipeline(
            "text2text-generation",
            model="vennify/t5-base-grammar-correction",
            device=0 if torch.cuda.is_available() else -1
        )

    def correct_sentence(self, sentence: str) -> str:
        """Correct a single sentence conservatively"""
        # Basic grammar/spelling correction
        matches = self.grammar_tool.check(sentence)
        corrected = language_tool_python.utils.correct(sentence, matches)

        # Light neural correction
        result = self.grammar_model(
            corrected,
            max_length=256,
            num_beams=3,
            temperature=0.3,  # Low temperature for minimal changes
            early_stopping=True
        )
        return result[0]['generated_text']

    def conservative_correction(self, text: str) -> str:
        """Process text while preserving original structure"""
        if not text.strip():
            return text

        # Split into sentences while preserving delimiters
        sentences = []
        current = ""
        for char in text:
            current += char
            if char in {'.', '!', '?'}:
                sentences.append(current)
                current = ""
        if current:
            sentences.append(current)

        # Correct each sentence individually
        corrected_sentences = []
        for sentence in sentences:
            if sentence.strip():
                corrected = self.correct_sentence(sentence)
                corrected_sentences.append(corrected)
            else:
                corrected_sentences.append(sentence)

        return ''.join(corrected_sentences)