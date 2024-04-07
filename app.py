from sense2vec import Sense2Vec
import os
from flask_limiter import Limiter
#from models.mcq import post_mca_questions
import csv
from flask import Flask, request, jsonify, render_template, session, redirect
import sqlite3
from utils.db_helper import UserDatabase
db =  UserDatabase(host="localhost", user="root", password="Sim@nta2023", database="MCQ")

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Set a secret key for session management
limiter = Limiter(app, default_limits=["10 per minute"])
def count_words(text):
    return len(text.split())
# Pass the enumerate function to Jinja2 environment
app.jinja_env.globals['enumerate'] = enumerate
# Add a before_request function for global rate limiting
@app.before_request
def before_request():
    # This decorator applies rate limiting to all routes
    limiter.limit("10/minute")(lambda: None)()

@app.before_request
def add_custom_header():
    print(request)
    if request.path != '/' and not request.path.endswith('.html'):
        referer = request.headers.get('Referer')
        #x_custom_passcode = request.headers.get("code")
        if referer is None or not referer.startswith('http://127.0.0.1:5000'):
        # if x_custom_passcode != 'abc':
            return jsonify({
                "success": False,
                "code": '400',
                "error": 'Unauthorized',
                "message": 'Missing Authorization',
                "traceback": '',
                "description": ''
            })
        
# Load Sense2Vec model
if os.path.exists("s2v_old"):
    s2v = Sense2Vec().from_disk('s2v_old')
    print("Sense2Vec model loaded successfully.")
else:
    print("Sense2Vec model file 's2v_old' not found.")
    s2v = None

@app.route("/")
def home():
	return render_template("cover.html")


# Route for login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = db.check_user(email, password)
        if user:
            session['logged_in'] = True
            return redirect("/index")
        else:
            return render_template("login.html", error="Invalid email or password.")
    return render_template("login.html", error="Invalid email or password.")

# Route for registration
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        user = db.check_user(email, password)
        if not user:
            db.insert_user(username,email, password)
            return redirect("/login")
        else:
            return render_template("register.html", error="Already Registered")
    return render_template("register.html")


@app.route("/index", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit():
    global parsed_questions 
    if request.method == "POST":
        text_input = request.form.get("textInput")
        print("Received text input:", text_input)

        if text_input:
            if count_words(text_input) > 500 and count_words(text_input) < 50:
                return jsonify({"error": "Word limit exceeded. Maximum 500 words allowed."}), 400

            #final_questions = post_mca_questions(text_input, s2v, num_questions=10)
            final_questions = [
            'What plays a pivotal role in shaping the trajectory of human development?(a)education(b)navigate life(c)confidence necessary(d)educationCorrect answer is : (d)',
            'What does Eric liu believe education is intrinsically linked to?(a)education(b)navigate life(c)navigate life(d)education significantly impacts healthCorrect answer is : (c)',
            'What does Eric liu believe is the most important factor in human development?(a)education(b)navigate life(c)confidence necessary(d)education significantly impacts healthCorrect answer is : (c)',
            'How does Eric liu feel about health and well being?(a)education(b)navigate life(c)education significantly impacts health(d)education significantly impacts healthCorrect answer is : (c)',
            'What does Eric liu believe education does?(a)education(b)empower individuals(c)confidence necessary(d)education significantly impacts healthCorrect answer is : (b)'
                ]
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
    

@app.route("/view_answer", methods=["POST"])
def view_answer():
    correct_answers = []
    for i, (question, options, correct_answer) in enumerate(parsed_questions, start=0):
        selected_option = request.form.get(f"question_{i}")
        if selected_option is not None:
            selected_answer = selected_option.split(')')[0].strip()  # Extract and strip the selected option
            if selected_answer.lower() in correct_answer.lower():
                result = "Correct"
            else:
                result = "Incorrect"
        else:
            selected_answer = "No option selected"
            result = "Incorrect"

        correct_answers.append((question, options, selected_answer, correct_answer, result))

    return render_template("view_answer.html", zipped_data=correct_answers)


if __name__ == '__main__':
    db.create_table()
    app.run(debug=True)
