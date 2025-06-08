import language_tool_python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Global variables for models
grammar_tool = None
tense_model = None
tense_tokenizer = None

def initialize_models():
    """Initialize all ML models at startup"""
    global grammar_tool, tense_model, tense_tokenizer

    print("Initializing Language Tool...")
    grammar_tool = language_tool_python.LanguageTool('en-US')

    print("Initializing T5 model...")
    model_name = "Vamsi/T5_Paraphrase_Paws"
    tense_tokenizer = T5Tokenizer.from_pretrained(model_name)
    tense_model = T5ForConditionalGeneration.from_pretrained(model_name)

def grammar_correction(text):
    """Correct grammar using LanguageTool"""
    if not grammar_tool:
        raise Exception("Grammar tool not initialized")

    matches = grammar_tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

def tense_correction(text):
    """Correct tense using T5 model"""
    if not tense_model or not tense_tokenizer:
        raise Exception("Tense correction models not initialized")

    input_text = "paraphrase: " + text + " </s>"
    encoding = tense_tokenizer.encode_plus(
        input_text,
        padding='max_length',
        return_tensors="pt",
        max_length=256,
        truncation=True
    )
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]

    outputs = tense_model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        max_length=256,
        num_return_sequences=1,
        num_beams=5,
        temperature=1.5
    )

    paraphrased = tense_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased

def correct_paragraph(text):
    """Complete text correction pipeline"""
    # Step 1: Grammar correction
    grammatically_correct = grammar_correction(text)
    print("After Grammar Correction:", grammatically_correct)

    # Step 2: Tense correction
    fully_corrected = tense_correction(grammatically_correct)
    print("After Grammar + Tense Correction:", fully_corrected)

    return fully_corrected