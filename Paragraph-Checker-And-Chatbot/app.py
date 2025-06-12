from flask import Flask, request, jsonify
from flask_cors import CORS
from paragraph_checker import ParagraphCorrector
from grammar_chatbot import GrammarChatbot
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
paragraph_service = ParagraphCorrector()
chatbot_service = GrammarChatbot()

@app.route('/correct_text', methods=['POST'])
def handle_paragraph():
    data = request.get_json()
    text = data.get("paragraph", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        conservative = paragraph_service.conservative_correction(text)
        fluent = paragraph_service.fluent_correction(text)

        return jsonify({
            "original": text,
            "conservative": conservative,
            "fluent": fluent
        })

    except Exception as e:
        logger.error(f"Correction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/chat', methods=['POST'])
def handle_chat():
    data = request.get_json()
    user_input = data.get("question", "")

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    try:
        response = chatbot_service.generate_response(user_input)
        return jsonify({
            "user_input": user_input,
            "response": response
        })

    except Exception as e:
        logger.error(f"Chatbot error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/reset', methods=['POST'])
def reset_conversation():
    try:
        chatbot_service.reset_conversation()
        return jsonify({"message": "Conversation reset successfully"})
    except Exception as e:
        logger.error(f"Reset error: {str(e)}")
        return jsonify({"error": "Failed to reset conversation"}), 500


@app.route('/health', methods=['GET'])
def health_check():
    try:
        _ = paragraph_service.conservative_correction("This is a test.")
        _ = chatbot_service.generate_response("Hello")
        return jsonify({
            "status": "healthy",
            "services": ["paragraph", "chat"]
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
