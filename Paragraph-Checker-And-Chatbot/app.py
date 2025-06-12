from flask import Flask, request, jsonify
from paragraph_checker import ParagraphCorrector
from grammar_chatbot import GrammarChatbot
import logging

app = Flask(__name__)

# Initialize services
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
        logger.error(f"Paragraph correction error: {str(e)}")
        return jsonify({
            "error": "Paragraph processing failed",
            "details": str(e)
        }), 500

@app.route('/chat', methods=['POST', 'GET'])  # Added GET method for testing
def handle_chat():
    """Endpoint for fluent conversational correction"""
    if request.method == 'POST':
        data = request.get_json()
        user_input = data.get('message', '').strip()
    else:  # GET method for testing
        user_input = request.args.get('message', '').strip()

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    try:
        response = chatbot_service.generate_response(user_input)
        return jsonify({
            "original_text": response["original_text"],
            "corrected_text": response["corrected_text"],
            "is_corrected": response["is_corrected"],
            "compliment": response["compliment"],
            "next_question": response["next_question"],
            "end_conversation": response["end_conversation"]
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
        response = chatbot_service.start_conversation()
        return jsonify(response)
    except Exception as e:
        logger.error(f"Start conversation error: {str(e)}")
        return jsonify({
            "error": "Failed to start conversation",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET', 'POST'])  # Added POST method for testing
def health_check():
    return jsonify({
        "status": "healthy",
        "services": ["paragraph", "chat"],
        "details": {
            "paragraph_service": "active",
            "chatbot_service": "active"
        }
    })

if __name__ == '__main__':
    logger.info("Starting grammar services...")
    app.run(host='0.0.0.0', port=5001, debug=True)