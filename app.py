from flask import Flask, request, jsonify
import requests
import os
from flask import Flask, render_template, request, make_response
import pandas as pd
import numpy as np          # For mathematical calculations
import matplotlib.pyplot as plt  # For plotting graphs
from datetime import datetime    # To access datetime
from pandas import Series        # To work on series
import csv
import io

import time

import warnings # To ignore the warnings warnings.filterwarnings("ignore")
from models.mcq import post_mca_questions
app = Flask(__name__)
# CORS(app)


def count_words(text):
    return len(text.split())

@app.route("/")
def index():
	return render_template("index.html")



@app.route("/submit", methods=["POST"])
def submit():
    if request.method == "POST":
        text_input = request.form.get("textInput")

        if text_input:
            if count_words(text_input) > 100:
                return jsonify({"error": "Word limit exceeded. Maximum 100 words allowed."}), 400
            
            final_questions = post_mca_questions(text_input, num_questions=10)
            # Split the text into individual questions
            questions = [q.strip() for q in final_questions.split(",")]
            parsed_questions = []
            for question in questions:
                parts = question.split("\n")
                q = parts[0].strip().replace('"', '')
                options = [o.strip()[3:] for o in parts[1:-1]]
                correct_answer = parts[-1].split(":")[-1].strip().replace('(', '').replace(')', '').replace('"', '')
                parsed_questions.append((q, options, correct_answer))

            # Render the result.html template with parsed questions
            return render_template("result.html", questions=parsed_questions)
        else:
            return "No data received."
    else:
        return "Method not allowed."


if __name__ == '__main__':
    app.run(debug=True)
