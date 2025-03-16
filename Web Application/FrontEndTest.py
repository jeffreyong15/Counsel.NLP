from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("webpage.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = {"message": f"Mock response for: {user_input}"}
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
