# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from askdocument import AskDocument
app = Flask(__name__)
CORS(app)
AD = AskDocument()

@app.route('/', methods = ['GET'])
def hello(name=None):
    print("LLM QnA Application")
    data = {"message": "LLM QnA Application"} 
    return jsonify(data)

@app.route('/topic', methods = ['POST'])
def add_or_create_topic():
    data = request.json
    AD.insert_or_fetch_embeddings(index_name=data['topic'])
    return jsonify({"status":"topic embedding ready"})

@app.route('/deletetopic', methods = ['POST'])
def delete_topic():
    # logic to check if you want to delete all topics
    data = request.json
    AD.delete_pinecone_index(index_name=data['topic'])
    return jsonify({"status":f"deleted topic(s): {data['topic']} "})

@app.route('/ask', methods = ['POST'])
def ask_question():
    # logic to check if you want to delete all topics
    data = request.json
    question = data['question']
    answer = AD.ask_and_get_answer(question)
    return jsonify({"answer":answer})

@app.route('/listtopics', methods = ['GET'])
def list_topics():
    print("getting list of topics")
    data = AD.get_list_of_indexes()
    return jsonify({"topics":data})

# set topic
@app.route('/settopic', methods = ['POST'])
def set_topic():
    data = request.json
    AD.set_topic(data['topic'])
    return jsonify({"status":f"topic set to {data['topic']} "})
