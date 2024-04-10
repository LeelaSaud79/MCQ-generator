import torch
import random
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
from flashtext import KeywordProcessor
from sklearn.metrics.pairwise import cosine_similarity
from sense2vec import Sense2Vec
import pke
import string
import pickle
import os
import time

# Define the device (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load or download and save summary model and tokenizer
summary_model_path = "t5_summary_model.pkl"
summary_tokenizer_path = "t5_summary_tokenizer.pkl"
if os.path.exists(summary_model_path) and os.path.exists(summary_tokenizer_path):
    with open(summary_model_path, 'rb') as f:
        summary_model = pickle.load(f)
    with open(summary_tokenizer_path, 'rb') as f:
        summary_tokenizer = pickle.load(f)
else:
    summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
    summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    with open(summary_model_path, 'wb') as f:
        pickle.dump(summary_model, f)
    with open(summary_tokenizer_path, 'wb') as f:
        pickle.dump(summary_tokenizer, f)

# Load or download and save question model and tokenizer
question_model_path = "t5_question_model.pkl"
question_tokenizer_path = "t5_question_tokenizer.pkl"
if os.path.exists(question_model_path) and os.path.exists(question_tokenizer_path):
    with open(question_model_path, 'rb') as f:
        question_model = pickle.load(f)
    with open(question_tokenizer_path, 'rb') as f:
        question_tokenizer = pickle.load(f)
else:
    question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
    question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
    with open(question_model_path, 'wb') as f:
        pickle.dump(question_model, f)
    with open(question_tokenizer_path, 'wb') as f:
        pickle.dump(question_tokenizer, f)

# Load or download and save sentence transformer model
sentence_transformer_model_path = "sentence_transformer_model.pkl"
if os.path.exists(sentence_transformer_model_path):
    with open(sentence_transformer_model_path, 'rb') as f:
        sentence_transformer_model = pickle.load(f)
else:
    sentence_transformer_model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v2")
    with open(sentence_transformer_model_path, 'wb') as f:
        pickle.dump(sentence_transformer_model, f)

# Move models to GPU if available
if device == "cuda":
    summary_model = summary_model.to(device)
    question_model = question_model.to(device)

def summarizer(text, model, tokenizer):
    text = text.strip().replace("\n", " ")
    text = "summarize: " + text
    max_len = 512
    encoding = tokenizer.encode_plus(text, max_length=max_len, pad_to_max_length=False, truncation=True, return_tensors="pt").to(device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    outs = model.generate(input_ids=input_ids,
                          attention_mask=attention_mask,
                          early_stopping=True,
                          num_beams=3,
                          num_return_sequences=1,
                          no_repeat_ngram_size=2,
                          min_length=75,
                          max_length=300)
    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    summary = dec[0]
    summary = summary.strip()
    return summary

def post_nouns_multipartite(content):
    out = []
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content, language='en')

        pos = {'PROPN', 'NOUN', 'ADJ', 'VERB', 'ADP', 'ADV', 'DET', 'CONJ', 'NUM', 'PRON', 'X'}

        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos)

        extractor.candidate_weighting(alpha=1.1, threshold=0.75, method='average')
        keyphrases = extractor.get_n_best(n=15)

        for val in keyphrases:
            out.append(val[0])
    except Exception as e:
        print("Error occurred:", e)
    return out

def post_keywords(original_text):
    keywords = post_nouns_multipartite(original_text)
    return keywords

def post_question(context, answer, model, tokenizer):
    text = "context: {} answer: {}".format(context, answer)
    encoding = tokenizer.encode_plus(text, max_length=384, pad_to_max_length=False, truncation=True, return_tensors="pt").to(device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    outs = model.generate(input_ids=input_ids,
                          attention_mask=attention_mask,
                          early_stopping=True,
                          num_beams=5,
                          num_return_sequences=1,
                          no_repeat_ngram_size=2,
                          max_length=72)
    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    question = dec[0].replace("question:", "")
    question = question.strip()
    return question

def sense2vec_post_words(word, s2v, top_n, question):
    output = []
    try:
        sense = s2v.get_best_sense(word, senses=["NOUN", "PERSON", "PRODUCT", "LOC", "ORG", "EVENT", "NORP", "WORK OF ART", "FAC", "GPE", "NUM", "FACILITY"])
        most_similar = s2v.most_similar(sense, n=top_n)
        output = [val[0] for val in most_similar]
    except:
        output = []
    threshold = 0.6
    final = [word]
    checklist = question.split()
    for x in output:
        if post_highest_similarity_score(final, x) < threshold and x not in final and x not in checklist:
            final.append(x)
    return final[1:]

def post_distractors(word, original_sentence, sense2vec_model, sentence_model, top_n, lambda_val):
    distractors = sense2vec_post_words(word, sense2vec_model, top_n, original_sentence)
    if len(distractors) == 0:
        return distractors
    distractors_new = [word.capitalize()]
    distractors_new.extend(distractors)
    embedding_sentence = original_sentence + " " + word.capitalize()
    keyword_embedding = sentence_model.encode([embedding_sentence])
    distractor_embeddings = sentence_model.encode(distractors_new)
    max_keywords = min(len(distractors_new), 5)
    filtered_keywords = mmr(keyword_embedding, distractor_embeddings, distractors_new, max_keywords, lambda_val)
    final = [word.capitalize()]
    for wrd in filtered_keywords:
        if wrd.lower() != word.lower():
            final.append(wrd.capitalize())
    final = final[1:]
    return final

def post_mca_questions(context: str, s2v, num_questions: int = 5):
    summarized_text = summarizer(context, summary_model, summary_tokenizer)
    imp_keywords = post_keywords(context)
    output_list = []
    
    for answer in imp_keywords:
        output = ""
        ques = post_question(summarized_text, answer, question_model, question_tokenizer)
        distractors = post_distractors(answer.capitalize(), ques, s2v, sentence_transformer_model, 40, 0.2)
        output = output + ques + "\n"
        if len(distractors) == 0:
            distractors = imp_keywords
        if len(distractors) > 0:
            random_integer = random.randint(0, 3)
            alpha_list = ['(a)', '(b)', '(c)', '(d)']
            for d, distractor in enumerate(distractors[:4]):
                if d == random_integer:
                    output = output + alpha_list[d] + answer + "\n"
                elif distractor!=answer:
                    output = output + alpha_list[d] + distractor + "\n"
                else:
                    try:
                        output = output + alpha_list[d] + distractors[5] + "\n"
                    except:
                        output = output + alpha_list[d] + "None Of the Above"+ "\n"
            output = output + "Correct answer is : " + alpha_list[random_integer] + "\n\n"
        output_list.append(output)
        if len(output_list) == num_questions:
            break
    mca_questions = output_list
    return mca_questions

# Function to get user input and generate multiple-choice questions
def generate_questions_from_user_input():
    # Load Sense2Vec model
    if os.path.exists("s2v_old"):
        s2v = Sense2Vec().from_disk('s2v_old')
        print("Sense2Vec model loaded successfully.")
    else:
        print("Sense2Vec model file 's2v_old' not found.")
        return
    
    # Prompt the user for input
    user_input = input("Enter the text: ")

    # Generate multiple-choice questions
    final_questions = post_mca_questions(user_input, s2v, num_questions=10)

    # Display the generated questions
    for i, question in enumerate(final_questions, 1):
        print(f"Question {i}:")
        print(question)


# Main function
if __name__ == "__main__":
    generate_questions_from_user_input()

