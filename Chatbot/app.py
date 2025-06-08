from flask import Flask, request, jsonify
from grammar_chatbot import GrammarChatbot

app = Flask(__name__)
chatbot = GrammarChatbot()  # Create an instance of the chatbot

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    bot_response = chatbot.generate_response(user_input)

    print(bot_response)

    return jsonify({
        "original_text": bot_response["original_text"],
        "corrected_text": bot_response["corrected_text"],
        "is_corrected": bot_response["is_corrected"],
        "compliment": bot_response["compliment"],
        "next_question": bot_response["next_question"],
        "end_conversation": bot_response["end_conversation"]
    })

@app.route('/correct', methods=['POST'])
def correct_text():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    corrected = chatbot.correct_grammar(text)
    print(corrected)
    return jsonify({
        "original_text": text,
        "corrected_text": corrected
    })

@app.route('/start', methods=['GET'])
def start_conversation():
    return jsonify(chatbot.start_conversation())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)