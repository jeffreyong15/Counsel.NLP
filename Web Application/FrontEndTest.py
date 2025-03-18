from flask import Flask, render_template, request, jsonify
from LLM import get_chatbot_response
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("webpage.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    chatbot_response = get_chatbot_response(user_input)

    return jsonify({"message": chatbot_response})

if __name__ == "__main__":
    app.run(debug=False)
