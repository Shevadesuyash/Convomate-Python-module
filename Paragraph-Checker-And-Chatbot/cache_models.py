import language_tool_python
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def pre_cache_models():
    """
    Downloads and caches all required models and dependencies.
    This script is run during the Docker build process.
    """
    print("Caching LanguageTool model...")
    try:
        # This will download and cache the LanguageTool server files
        language_tool_python.LanguageTool('en-US')
        print("LanguageTool model cached successfully.")
    except Exception as e:
        print(f"Failed to cache LanguageTool: {e}")

    print("\nCaching Hugging Face models...")
    models_to_cache = [
        "vennify/t5-base-grammar-correction",
        "humarin/chatgpt_paraphraser_on_T5_base"
    ]

    for model_name in models_to_cache:
        try:
            print(f"Caching {model_name}...")
            # Cache both tokenizer and model files
            AutoTokenizer.from_pretrained(model_name)
            AutoModelForSeq2SeqLM.from_pretrained(model_name)
            print(f"{model_name} cached successfully.")
        except Exception as e:
            print(f"Failed to cache {model_name}: {e}")

    print("\nAll models have been cached.")

if __name__ == "__main__":
    pre_cache_models()