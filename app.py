#summarizer

from flask import Flask, request, jsonify
from summarizer_logic import summarize_text

app = Flask(__name__)

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get("text", "")
    
    summarized_text = summarize_text(text)

    return jsonify({
        "original_text": text,
        "summarized_text": summarized_text
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
