from flask import Flask, request, jsonify, render_template, session, redirect
from models.mcq import post_mca_questions
from sense2vec import Sense2Vec
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Set a secret key for session management

def count_words(text):
    return len(text.split())

# Load Sense2Vec model
if os.path.exists("s2v_old"):
    s2v = Sense2Vec().from_disk('s2v_old')
    print("Sense2Vec model loaded successfully.")
else:
    print("Sense2Vec model file 's2v_old' not found.")
    s2v = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit():
    if request.method == "POST":
        text_input = request.form.get("textInput")
        print("Received text input:", text_input)

        if text_input:
            if count_words(text_input) > 100:
                return jsonify({"error": "Word limit exceeded. Maximum 100 words allowed."}), 400

            final_questions = post_mca_questions(text_input, s2v, num_questions=5)
            print("Generated MCQs:", final_questions)

            # Store final_questions in session
            session['final_questions'] = final_questions

            parsed_questions = []
            for question_data in final_questions:
                question_parts = question_data.split("?")
                question = question_parts[0].strip() + "?"

                options_parts = question_parts[1].split("Correct answer is : ")
                options = options_parts[0].strip().split("(")[1:]

                correct_answer = options_parts[1].strip().split(":")[-1].strip()

                parsed_questions.append((question, options, correct_answer))

            return render_template("result.html", questions=parsed_questions)
        else:
            return "No data received."
    else:
        return "Method not allowed."

@app.route("/view_answer")
def view_answer():
    # Retrieve final_questions from session
    final_questions = session.get('final_questions', None)
    if final_questions:
        # Split each element into a question and a correct answer
        questions = [question_answer.split('\n') for question_answer in final_questions]
        return render_template("view_answer.html", questions=questions)
    else:
        return "No MCQs generated."

@app.route("/submit_answers", methods=["POST"])
def submit_answers():
    return redirect("/view_answer")

if __name__ == '__main__':
    app.run(debug=True)
