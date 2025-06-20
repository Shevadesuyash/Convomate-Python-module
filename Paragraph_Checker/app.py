from flask import Flask, request, jsonify
from paragraph_checker import ParagraphCorrector

app = Flask(__name__)
corrector = ParagraphCorrector()

@app.route('/correct_text', methods=['POST'])
def correct_text():
    data = request.get_json()

    # Get paragraph from request (compatible with Spring DTO)
    text = data.get('paragraph', '').strip()

    if not text:
        return jsonify({
            "error": "No paragraph provided"
        }), 400

    try:
        # Get correction (using default aggressive=False)
        corrected = corrector.correct_paragraph(text)

        # Return response matching Spring DTO structure
        return jsonify({
            "original_text": text,
            "grammar_corrected": corrected  # Changed from "corrected_text"
        })

    except Exception as e:
        return jsonify({
            "error": "An error occurred while processing the text",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)