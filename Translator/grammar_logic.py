import language_tool_python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load LanguageTool for grammar and spelling correction
tool = language_tool_python.LanguageTool('en-US')



# Load T5 model for tense transformation
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)



# Function to correct grammar and spelling
def correct_grammar(text):
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

# Function to change tense using T5 model
def change_tense(text, target_tense="past"):
    input_text = f"change tense to {target_tense}: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    transformed_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return transformed_text
