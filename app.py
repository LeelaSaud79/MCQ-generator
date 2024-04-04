from flask import Flask, request, jsonify
#import requests
#import os
from flask import Flask, render_template, request, make_response
#import pandas as pd
#import numpy as np          # For mathematical calculations
#import matplotlib.pyplot as plt  # For plotting graphs
#from datetime import datetime    # To access datetime
#from pandas import Series        # To work on series
#import csv
#import io

#import time

#import warnings # To ignore the warnings warnings.filterwarnings("ignore")
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
    global parsed_questions 
    if request.method == "POST":
        text_input = request.form.get("textInput")

        if text_input:
            if count_words(text_input) > 100:
                return jsonify({"error": "Word limit exceeded. Maximum 100 words allowed."}), 400
            
            #final_questions = ['What plays a pivotal role in shaping the trajectory of human development?(a)education(b)navigate life(c)confidence necessary(d)educationCorrect answer is : (d)', 'What does Eric liu believe education is intrinsically linked to?(a)education(b)navigate life(c)navigate life(d)education significantly impacts healthCorrect answer is : (c)', 'What does Eric liu believe is the most important factor in human development?(a)education(b)navigate life(c)confidence necessary(d)education significantly impacts healthCorrect answer is : (c)', 'How does Eric liu feel about health and well being?(a)education(b)navigate life(c)education significantly impacts health(d)education significantly impacts healthCorrect answer is : (c)', 'What does Eric liu believe education does?(a)education(b)empower individuals(c)confidence necessary(d)education significantly impacts healthCorrect answer is : (b)']
            final_questions = post_mca_questions(text_input, num_questions=10)
            parsed_questions = []
            for question_data in final_questions:
                # Split the question data into question and answer parts
                parts = question_data.rsplit('Correct answer is : ', 1)
                question_with_options = parts[0]
                correct_answer = parts[1].strip()

                # Extract question and options
                question = question_with_options.split('(', 1)[0].strip()
                options = [opt.strip() for opt in question_with_options.split(')')[0].split('(', 1)[1].split(')') if opt.strip()]

                parsed_questions.append((question, options, correct_answer))

            return render_template("result.html", questions=parsed_questions)
        else:
            return "No data received."
    else:
        return "Method not allowed."



@app.route("/view_answer", methods=["GET", "POST"])
def view_answer():
    global parsed_questions  # Access the global variable
    return render_template("view_answer.html", questions=parsed_questions)


if __name__ == '__main__':
    app.run(debug=True)