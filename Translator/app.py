from flask import Flask, request, jsonify
from translator_logic import detect_language, translate_text

app = Flask(__name__)

@app.route('/translate', methods=['POST'])
def translate():
    # Log the incoming request payload
    data = request.get_json()
    print("Received request payload:", data)  # Debugging

    from_language = data.get('from_language', 'auto')  # Default to 'auto' for automatic detection
    to_language = data.get('to_language', 'en')  # Default to English
    text_to_translate = data.get('text_to_translate', '')

    if not text_to_translate:
        print("Error: No text provided for translation")  # Debugging
        return jsonify({"error": "No text provided for translation"}), 400

    # Log the translation parameters
    print(f"Translating from {from_language} to {to_language}: {text_to_translate}")  # Debugging

    # Perform translation
    translated_text, pronunciation = translate_text(from_language, to_language, text_to_translate)

    if translated_text is None:
        print("Translation error:", pronunciation)  # Debugging
        return jsonify({"error": pronunciation}), 500

    # Prepare the response
    response = {
        'translatedText': translated_text,
        'pronunciation': pronunciation,  # Include pronunciation in the response
        'fromLanguage': from_language,
        'toLanguage': to_language
    }

    # Log the outgoing response
    print("Sending response:", response)  # Debugging

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)