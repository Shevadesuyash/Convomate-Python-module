from googletrans import Translator


def detect_language(text_to_detect):
    try:
        translator = Translator()
        detected_language = translator.detect(text_to_detect).lang
        return detected_language
    except Exception as e:
        return f"Language detection error: {str(e)}"

def translate_text(from_language, to_language, text_to_translate):
    try:
        translator = Translator()

        if from_language == "auto":
            # Detect the input language
            from_language = detect_language(text_to_translate)

        # Perform translation
        translation = translator.translate(text_to_translate, src=from_language, dest=to_language)

        translated_text = translation.text

        # Get pronunciation for English or original language
        pronunciation = None
        if from_language == "en":
            # If translating from English, get pronunciation of the translated text
            pronunciation = translation.pronunciation
        elif to_language == "en":
            # If translating to English, get pronunciation of the original text
            original_pronunciation = translator.translate(text_to_translate, src=from_language, dest="en").pronunciation
            pronunciation = original_pronunciation
        else:
            # For other cases, try to get pronunciation of the translated text
            pronunciation = translation.pronunciation

        return translated_text, pronunciation
    except Exception as e:
        return None, f"Translation error: {str(e)}"