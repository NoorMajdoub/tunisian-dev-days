
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/chatbot")
def chatbot():
    return {"members": ["Yourr medical report indicates a high risk of Cardiomegaly Disease , the use of Ibuprofen might have negative impact on your health, please refer to your doctor before consuming Ibuprofen"]}
@app.route("/diagnosis",methods=["POST"])
def diagnosis():
    data = request.get_json()
    res = "Response from diagnosis"
    user_input = data.get("input")
    return {"diagnosis": user_input}
if __name__ == "__main__":
    app.run(debug=True)
