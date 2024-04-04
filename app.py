from flask import Flask, request, jsonify
import requests

from mcq import post_mca_questions
app = Flask(__name__)
# CORS(app)


@app.route('/generate_questions', methods=['POST'])
def generate_questions():

    data = request.json
    text = data.get('text')
    num_questions = data.get('num_questions', 10)  
    final_questions = post_mca_questions(text, num_questions=num_questions)
    return jsonify({'questions': final_questions})

if __name__ == '__main__':
    app.run(debug=True)
