import language_tool_python
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional
import torch

class ParagraphCorrector:
    def __init__(self):
        """Initialize all correction models"""
        print("Initializing correction models...")

        # Initialize LanguageTool for grammar/spelling
        self.grammar_tool = language_tool_python.LanguageTool('en-US')

        # Initialize neural grammar correction model
        self.grammar_model = pipeline(
            "text2text-generation",
            model="vennify/t5-base-grammar-correction",
            device=0 if torch.cuda.is_available() else -1
        )

        # Initialize paraphrase model for tense/style (with lower temperature)
        self.paraphrase_tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
        self.paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

        print("All models loaded successfully!")

    def grammar_correction(self, text: str) -> str:
        """Correct grammar and spelling using LanguageTool"""
        matches = self.grammar_tool.check(text)
        return language_tool_python.utils.correct(text, matches)

    def neural_grammar_correction(self, text: str) -> str:
        """Correct grammar using neural model"""
        result = self.grammar_model(
            text,
            max_length=256,
            num_beams=5,
            repetition_penalty=1.0,
            early_stopping=True
        )
        return result[0]['generated_text']

    def paraphrase_text(self, text: str, temperature: float = 0.7) -> str:
        """Paraphrase text with controlled randomness for tense/style"""
        input_ids = self.paraphrase_tokenizer(
            f"paraphrase: {text}",
            return_tensors="pt",
            padding="longest",
            max_length=256,
            truncation=True
        ).input_ids

        outputs = self.paraphrase_model.generate(
            input_ids,
            temperature=temperature,
            repetition_penalty=1.0,
            num_return_sequences=1,
            do_sample=True,
            max_length=256
        )

        return self.paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def correct_paragraph(self, text: str, aggressive: bool = False) -> str:
        """
        Complete paragraph correction pipeline
        :param text: Input text to correct
        :param aggressive: Whether to use more aggressive neural corrections
        :return: Corrected text
        """
        print("\nOriginal text:", text)

        # Step 1: Basic grammar/spelling correction
        corrected = self.grammar_correction(text)
        print("\nAfter grammar correction:", corrected)

        # Step 2: Neural grammar correction (more comprehensive)
        if aggressive:
            corrected = self.neural_grammar_correction(corrected)
            print("\nAfter neural grammar correction:", corrected)

            # Re-check with LanguageTool
            corrected = self.grammar_correction(corrected)
            print("\nAfter grammar re-check:", corrected)

        # Step 3: Tense/style correction (carefully applied)
        paraphrased = self.paraphrase_text(corrected)
        print("\nAfter tense/style adjustment:", paraphrased)

        # Final grammar check
        final_output = self.grammar_correction(paraphrased)
        print("\nFinal corrected text:", final_output)

        return final_output

# Example usage
if __name__ == "__main__":
    corrector = ParagraphCorrector()

    sample_text = """
    He go to school everyday but yesterday he not goes because he was sick. 
    His teacher give him many homeworks that he must to complete before next class.
    """

    corrected_text = corrector.correct_paragraph(sample_text, aggressive=True)
    print("\n=== Final Result ===")
    print("Original:", sample_text)
    print("Corrected:", corrected_text)