# app.py
from flask import Flask, request, jsonify
from paragraph_checker import correct_paragraph, initialize_models
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/correct_text', methods=['POST'])
def api_correct_text():
    data = request.get_json()
    text = data.get("paragraph", "").strip()
    aggressive = data.get("aggressive", False)

    if not text:
        return jsonify({"error": "No text provided"}), 400

    logger.info(f"Processing text (length: {len(text)}, aggressive: {aggressive})")

    try:
        # Get fully corrected text
        fully_corrected = correct_paragraph(text, aggressive=aggressive)

        response = {
            "original_text": text,
            "corrected_text": fully_corrected,
            "original_length": len(text),
            "corrected_length": len(fully_corrected)
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing text: {str(e)}", exc_info=True)
        return jsonify({
            "error": "An error occurred while processing the text",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    # Initialize models at startup
    logger.info("Loading ML models...")
    try:
        initialize_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}", exc_info=True)
        raise e

    app.run(host="0.0.0.0", port=5001, debug=True)