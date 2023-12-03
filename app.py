from flask import Flask, jsonify, request, render_template
from flask_restful import reqparse
import os
import pinecone
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

parser = reqparse.RequestParser()
parser.add_argument('query')

hf_api_token = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# Replicate API token
replicate_api_token = os.environ['REPLICATE_API_TOKEN']

# Initialize Pinecone
pinecone_api_token = os.environ['PINECONE_API_TOKEN']
pinecone_index_name = os.environ['PINECONE_INDEX_NAME']
pinecone_env = os.environ['PINECONE_ENV_NAME']

pinecone.init(api_key=pinecone_api_token, environment=pinecone_env)

def create_vectordb(filepath):
    # Load and preprocess the PDF document
    loader = PyPDFLoader(filepath)
    documents = loader.load()

    # Split the documents into smaller chunks for processing
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Use HuggingFace embeddings for transforming text into numerical vectors
    embeddings = HuggingFaceEmbeddings()

    # Set up the Pinecone vector database
    index_name = pinecone_index_name
    index = pinecone.Index(index_name)
    vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)
    return vectordb


def init_qa_model(filepath):
    # Initialize Replicate Llama2 Model
    llm = Replicate(
        model= "meta/llama-2-13b-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5", #"a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
        model_kwargs={"temperature": 0.9, "max_length": 3000}
    )

    vectordb = create_vectordb(filepath)

    # Set up the Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectordb.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True
    )

    return qa_chain


def init_chat_model():

    template = """You are an Assistant named as ResBot. Your job is to answer questions only related to Copyright and IP rights of artworks, Copyright and IP rights of artists, Copyright and IP rights of \
    NFT arts and Copyright and IP rights of generative AI arts. If you receive any question outside of these options, you will politely refuse to answer and redirect the user to ask questions related to \
    these above mentioned options. Always answer in just one paragraph. Do not return {history} as part of the answer. Now answer the following question.

    {history}
    Human: {human_input}
    Assistant:"""
    llm = Replicate(
        model= "meta/llama-2-13b-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5", #"a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
        model_kwargs={"temperature": 0.9, "max_length": 3000}
    )
    prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=False, memory=ConversationBufferWindowMemory(k=2), llm_kwargs={"max_length": 4096})

    return llm_chain

@app.route('/', methods=['GET'])
def sayHello():
    return jsonify({'message': 'Welcome to The Resolutio Official Bot😍'})

@app.route('/bot',  methods=['POST'])
def run_pipeline():
    global qa_chain
    global chat_history
    global llm_chain 

    chat_history = []
    body = request.get_json()

    # Access specific details from the request body
    query = body['message']
    category = body['category'] 
    userId = body['userId']
    timeStamp = body['timeStamp']

    if category.lower() == 'general ip queries' or category == "":        
        llm_chain  = init_chat_model()

        result = llm_chain.predict(human_input=query)       
        output = {"query": query, "result": result, "userId": userId, "timeStamp": timeStamp}
        chat_history.append(output)        
        return jsonify(output)

    if category.lower == 'azuki license':
        qa_chain = init_qa_model('./files/azuki_license_man.pdf')        

    elif category.lower == 'viral public license':
        qa_chain = init_qa_model('./files/Viral_Public_License.pdf')        

    elif category.lower() == 'byac License':
        qa_chain = init_qa_model('./files/BAYC.pdf')

    elif category.lower() == 'mayc license':
        qa_chain = init_qa_model('./files/MAYC.pdf')        
    
    result = qa_chain({'question': query, 'chat_history': chat_history})
    output = {"query": query, "result": result['answer'], "userId": userId, "timeStamp": timeStamp}
    chat_history.append(output)        
    return jsonify(output)

if __name__ == "__main__":
    app.run(port=6321, debug=True)
