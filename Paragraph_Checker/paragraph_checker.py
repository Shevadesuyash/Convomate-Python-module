import language_tool_python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch # Ensure torch is imported for device management

# Global variables for models
grammar_tool = None
paraphrase_model = None # Renamed for clarity
paraphrase_tokenizer = None

def initialize_models():
    """Initialize all ML models at startup"""
    global grammar_tool, paraphrase_model, paraphrase_tokenizer

    print("Initializing Language Tool...")
    # LanguageTool might need to download resources the first time
    grammar_tool = language_tool_python.LanguageTool('en-US')
    print("Language Tool initialized.")

    print("Initializing T5 Paraphrase model...")
    model_name = "Vamsi/T5_Paraphrase_Paws"
    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paraphrase_tokenizer = T5Tokenizer.from_pretrained(model_name)
    paraphrase_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    print(f"T5 Paraphrase model initialized on device: {device}.")

def grammar_correction(text):
    """Correct grammar using LanguageTool"""
    if not grammar_tool:
        # This shouldn't happen if initialize_models is called at startup
        # but good for robust error handling during development.
        raise Exception("Grammar tool not initialized. Call initialize_models() first.")

    matches = grammar_tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

def paraphrase_text(text): # Renamed for clarity
    """Paraphrase text using T5 model"""
    if not paraphrase_model or not paraphrase_tokenizer:
        raise Exception("Paraphrase models not initialized. Call initialize_models() first.")

    device = paraphrase_model.device # Get the device the model is on

    input_text = "paraphrase: " + text + " </s>"
    encoding = paraphrase_tokenizer.encode_plus(
        input_text,
        padding='max_length',
        return_tensors="pt",
        max_length=256,
        truncation=True
    )
    # Move tensors to the correct device
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    # Add no_grad for inference
    with torch.no_grad():
        outputs = paraphrase_model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            max_length=256,
            num_return_sequences=1,
            num_beams=5,
            temperature=1.5 # Temperature can make output more creative/diverse
        )

    paraphrased = paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased

def correct_paragraph(text):
    """Complete text correction pipeline"""
    # Step 1: Grammar correction
    grammatically_correct = grammar_correction(text)
    print("After Grammar Correction:", grammatically_correct)

    # Step 2: Paraphrasing (which might implicitly correct some tense/structure)
    fully_corrected = paraphrase_text(grammatically_correct)
    print("After Grammar + Paraphrasing:", fully_corrected)

    return fully_corrected