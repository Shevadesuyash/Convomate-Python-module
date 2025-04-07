from flask import Flask, request, jsonify
from grammar_logic import correct_grammar

app = Flask(__name__)

@app.route('/correct_text', methods=['POST'])
def correct_text():
    data = request.get_json()
    text = data.get("paragraph", "")

    print("test to correct  : ",text)

    corrected_text = correct_grammar(text)

    print("test to correct  : ",corrected_text)

    return jsonify({
        "original_text": text,
        "grammar_corrected": corrected_text
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
