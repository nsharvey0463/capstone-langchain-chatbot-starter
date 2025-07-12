import os

from flask import Flask, render_template
from flask import request, jsonify, abort
from langchain.llms import Cohere
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

def load_db():
    try:
        embeddings = CohereEmbeddings(cohere_api_key=os.environ["COHERE_API_KEY"])
        vectordb = Chroma(persist_directory='db', embedding_function=embeddings)
        qa = RetrievalQA.from_chain_type(
            llm=Cohere(),
            chain_type="refine",
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )
        return qa
    except Exception as e:
        print("Error:", e)

qa = load_db()

def answer_from_knowledgebase(message):
    res = qa({"query": message})
    return res['result']

app = Flask(__name__)
cohere_api_key = os.environ.get("COHERE_API_KEY")

if not cohere_api_key:
    raise ValueError("COHERE_API_KEY is not set in the environment variables.") 

# Initialize Cohere LLM
llm = Cohere(cohere_api_key=cohere_api_key)

def search_knowledgebase(message):
    res = qa({"query": message})
    sources = ""
    for count, source in enumerate(res['source_documents'],1):
        sources += "Source " + str(count) + "\n"
        sources += source.page_content + "\n"
    
    return sources

def answer_as_chatbot(message):
    # Use Cohere LLM to generate a response
    response = llm(message)
    return response

@app.route('/kbanswer', methods=['POST'])
def kbanswer():
    data = request.get_json()
    message = data.get('message', '')
    response = answer_from_knowledgebase(message)
    return jsonify({'message': response}), 200

@app.route('/search', methods=['POST'])
def search():    
    data = request.get_json()
    message = data.get('message', '')
    sources = search_knowledgebase(message)
    return jsonify({'sources': sources}), 200

@app.route('/answer', methods=['POST'])
def answer():
    message = request.json['message']
    response_message = answer_as_chatbot(message)
    return jsonify({'message': response_message}), 200

@app.route("/")
def index():
    return render_template("index.html", title="")

if __name__ == "__main__":
    app.run()