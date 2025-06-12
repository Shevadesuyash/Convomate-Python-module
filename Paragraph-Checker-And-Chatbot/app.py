from flask import Flask, request, jsonify
from paragraph_checker import ParagraphCorrector
from grammar_chatbot import GrammarChatbot
import logging

app = Flask(__name__)

# Initialize correction services
paragraph_service = ParagraphCorrector()
chatbot_service = GrammarChatbot()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/correct_text', methods=['POST'])
def handle_paragraph():
    """Endpoint for conservative paragraph correction"""
    data = request.get_json()
    text = data.get('paragraph', '').strip()

    if not text:
        return jsonify({"error": "No paragraph provided"}), 400

    try:
        corrected = paragraph_service.conservative_correction(text)
        return jsonify({
            "original_text": text,
            "grammar_corrected": corrected
        })
    except Exception as e:
        logger.error(f"Paragraph correction failed: {str(e)}")
        return jsonify({
            "error": "Paragraph processing failed",
            "details": str(e)
        }), 500

@app.route('/chat', methods=['POST'])
def handle_chat():
    """Endpoint for fluent conversational correction"""
    data = request.get_json()
    user_input = data.get('message', '').strip()

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    try:
        bot_response = chatbot_service.generate_response(user_input)
        logger.info(f"Chat response: {bot_response}")

        return jsonify({
            "original_text": bot_response["original_text"],
            "corrected_text": bot_response["corrected_text"],
            "is_corrected": bot_response["is_corrected"],
            "compliment": bot_response["compliment"],
            "next_question": bot_response["next_question"],
            "end_conversation": bot_response["end_conversation"]
        })
    except Exception as e:
        logger.error(f"Chatbot error: {str(e)}")
        return jsonify({
            "error": "Chat processing failed",
            "details": str(e)
        }), 500

@app.route('/start', methods=['GET'])
def start_conversation():
    try:
        start_response = chatbot_service.start_conversation()
        return jsonify(start_response)
    except Exception as e:
        logger.error(f"Start conversation error: {str(e)}")
        return jsonify({
            "error": "Failed to start conversation",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "services": {
            "paragraph_correction": "active",
            "chatbot": "active"
        }
    })

if __name__ == '__main__':
    logger.info("Starting combined grammar services...")
    app.run(host='0.0.0.0', port=5001)