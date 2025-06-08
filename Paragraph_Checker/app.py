from flask import Flask, request, jsonify
from paragraph_checker import correct_paragraph

app = Flask(__name__)

@app.route('/correct_text', methods=['POST'])
def correct_text():
    data = request.get_json()
    text = data.get("paragraph", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    print("Original Text:", text)

    try:
        # Get fully corrected text
        fully_corrected = correct_paragraph(text)

        return jsonify({
            "original_text": text,
            "corrected_text": fully_corrected
        })

    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return jsonify({
            "error": "An error occurred while processing the text",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    # Initialize models at startup
    from paragraph_checker import initialize_models
    print("Loading ML models...")
    try:
        initialize_models()
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {str(e)}")

    app.run(host="0.0.0.0", port=5001, debug=True)