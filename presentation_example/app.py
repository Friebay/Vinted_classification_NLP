from flask import Flask, render_template, request, jsonify
import sys
import os

from get_nn_model_results import predict_text

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    
    title = data.get("title", "")
    description = data.get("description", "")

    print(title)
    print(description)
    
    try:
        classified_category = predict_text(title, description)
        return jsonify({"success": True, "category": classified_category})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)