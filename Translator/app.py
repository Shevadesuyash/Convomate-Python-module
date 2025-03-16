from flask import Flask, request, jsonify
from grammar_logic import correct_grammar, change_tense

app = Flask(__name__)

@app.route('/correct_text', methods=['POST'])
def correct_text():
    data = request.get_json()
    text = data.get("text", "")
    target_tense = data.get("target_tense", "past")

    corrected_text = correct_grammar(text)
    tense_corrected_text = change_tense(corrected_text, target_tense)

    return jsonify({
        "original_text": text,
        "grammar_corrected": corrected_text,
        "tense_corrected": tense_corrected_text,
        "target_tense": target_tense
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
