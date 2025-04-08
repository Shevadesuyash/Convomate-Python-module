
import language_tool_python
from transformers import T5ForConditionalGeneration, T5Tokenizer

def grammar_correction(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

def tense_correction(text):
    model_name = "Vamsi/T5_Paraphrase_Paws"
    tokenizer = T5Tokenizer.from_pretrained(model_name, quiet=True)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    input_text = "paraphrase: " + text + " </s>"
    encoding = tokenizer.encode_plus(
        input_text, padding='max_length', return_tensors="pt", max_length=256, truncation=True
    )
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        max_length=256,
        num_return_sequences=1,
        num_beams=5,
        temperature=1.5
    )

    paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased

def correct_paragraph(text):
    grammatically_correct = grammar_correction(text)
    fully_corrected = tense_correction(grammatically_correct)

    print("Original Text:\n", text)
    print("\nAfter Grammar Correction:\n", grammatically_correct)
    print("\nAfter Grammar + Tense Correction:\n", fully_corrected)

    return fully_corrected


if __name__ == "__main__":
    paragraph = input("Enter a paragraph: ")
    correct_paragraph(paragraph)