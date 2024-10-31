
from flask import Flask, request, jsonify
from flask_cors import CORS
from combinedmodel import *
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route("/diagnosis",methods=["POST"])
def diagnosis():
    data = request.get_json()
    res = "Response from diagnosis"
    user_input = data.get("input")
    res=combinedresult(user_input)
    return {"diagnosis": res}
if __name__ == "__main__":
    app.run(debug=True)
