import os
import requests
import asyncio
import time
import re
from flask import Flask, request, jsonify

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Pre-shared bearer token for simple authorization
BEARER_TOKEN = "3ca0894d22ac6bf6daf7d8323b1e77d69241f8b2810b9bee667a0a14969ffb48"

# Rate limiting variables
last_request_time = 0
min_request_interval = 1  # Reduced to 1 second for Mistral (better rate limits)

def setup_event_loop():
    """Set up event loop for async operations."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def rate_limit():
    """Simple rate limiting to avoid hitting API quotas."""
    global last_request_time
    current_time = time.time()
    time_since_last = current_time - last_request_time
    
    if time_since_last < min_request_interval:
        sleep_time = min_request_interval - time_since_last
        time.sleep(sleep_time)
    
    last_request_time = time.time()

def simple_text_search(question, documents, max_results=3):
    """Simple text-based search when AI API is not available."""
    question_lower = question.lower()
    question_words = set(re.findall(r'\b\w+\b', question_lower))
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom', 'this', 'that', 'these', 'those'}
    question_words = question_words - stop_words
    
    scored_docs = []
    for doc in documents:
        content_lower = doc.page_content.lower()
        score = 0
        
        # Score based on word matches
        for word in question_words:
            if word in content_lower:
                score += 1
        
        # Bonus for exact phrase matches
        for word in question_words:
            if word in content_lower:
                score += 0.5
        
        if score > 0:
            scored_docs.append((score, doc))
    
    # Sort by score and return top results
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:max_results]]

def generate_simple_answer(question, context):
    """Generate a simple answer based on context when AI is not available."""
    if not context:
        return "The answer is not available in the provided document."
    
    # Simple keyword-based answer generation
    question_lower = question.lower()
    context_lower = context.lower()
    
    # Look for sentences that contain question keywords
    sentences = re.split(r'[.!?]+', context)
    relevant_sentences = []
    
    question_words = set(re.findall(r'\b\w+\b', question_lower))
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom', 'this', 'that', 'these', 'those'}
    question_words = question_words - stop_words
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = 0
        for word in question_words:
            if word in sentence_lower:
                score += 1
        if score > 0:
            relevant_sentences.append((score, sentence.strip()))
    
    if relevant_sentences:
        relevant_sentences.sort(key=lambda x: x[0], reverse=True)
        return relevant_sentences[0][1]
    else:
        return "The answer is not available in the provided document."

# --- Helper Functions for the RAG Pipeline ---

def download_pdf(url: str, save_path: str) -> bool:
    """Downloads a PDF from a URL and saves it locally."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {e}")
        return False

def get_text_chunks_from_pdf(pdf_path: str) -> list:
    """Loads text from a PDF and splits it into manageable chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    # Reduce chunk size to minimize token usage
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_vector_store(text_chunks: list):
    """Creates a FAISS vector store from text chunks using Mistral's embedding model."""
    # Set up event loop for async operations
    setup_event_loop()
    
    try:
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        return vector_store, True  # Return success flag
    except Exception as e:
        print(f"Failed to create vector store with Mistral embeddings: {e}")
        return text_chunks, False  # Return documents directly

def get_conversational_chain():
    """Creates a question-answering chain with a custom prompt and a Mistral LLM."""
    prompt_template = """
    Answer the question based on the provided context. Keep your answer concise and relevant.
    If the answer is not in the provided context, just say, "The answer is not available in the provided document".
    Do not provide a wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    try:
        model = ChatMistralAI(model="mistral-large-latest", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = prompt | model | StrOutputParser()
        return chain, True  # Return success flag
    except Exception as e:
        print(f"Failed to create Mistral AI chain: {e}")
        return None, False

# --- Flask API Endpoint ---

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint for deployment platforms."""
    return jsonify({
        "status": "healthy",
        "message": "HackRx RAG API is running",
        "endpoint": "/hackrx/run"
    })

@app.route("/test", methods=["GET"])
def test_endpoint():
    """Test endpoint to verify the app is working."""
    return jsonify({
        "message": "HackRx API is working!",
        "timestamp": time.time()
    })

@app.route("/hackrx/run", methods=["POST"])
def process_request():
    """
    The main endpoint to receive document URL and questions, and return answers.
    """
    # 1. Authorization Check
    auth_header = request.headers.get("Authorization")
    if not auth_header or f"Bearer {BEARER_TOKEN}" != auth_header:
        return jsonify({"error": "Unauthorized"}), 401

    # 2. Request Body Validation
    try:
        data = request.get_json()
        doc_url = data["documents"]
        questions = data["questions"]
        if not doc_url or not isinstance(questions, list):
            raise ValueError("Invalid request format.")
        
        # Limit number of questions to avoid rate limits
        if len(questions) > 10:  # Increased limit for Mistral
            return jsonify({"error": "Maximum 10 questions allowed per request to avoid rate limits."}), 400
            
    except (ValueError, KeyError):
        return jsonify({"error": "Bad Request: 'documents' URL and a list of 'questions' are required."}), 400

    # 3. RAG Pipeline Execution
    pdf_save_path = "temp_policy.pdf"
    try:
        # Set up event loop for async operations
        setup_event_loop()
        
        # Download the document
        if not download_pdf(doc_url, pdf_save_path):
             return jsonify({"error": "Failed to download the document from the provided URL."}), 500

        # Process the document to create a QA system
        text_chunks = get_text_chunks_from_pdf(pdf_save_path)
        vector_store, ai_available = get_vector_store(text_chunks)
        chain, chain_available = get_conversational_chain()

        # Answer each question with rate limiting
        answers = []
        for i, question in enumerate(questions):
            try:
                # Rate limiting between requests
                if i > 0 and ai_available:
                    rate_limit()
                
                if ai_available and chain_available:
                    # Use AI-based search and answer generation
                    try:
                        docs = vector_store.similarity_search(question, k=3)
                        context = "\n\n".join([doc.page_content for doc in docs])
                        response = chain.invoke({"context": context, "question": question})
                        answers.append(response)
                    except Exception as e:
                        error_msg = str(e)
                        if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                            # Fallback to simple text search
                            docs = simple_text_search(question, text_chunks)
                            context = "\n\n".join([doc.page_content for doc in docs])
                            answer = generate_simple_answer(question, context)
                            answers.append(f"{answer} (Note: Using fallback mode due to API limits)")
                        else:
                            answers.append(f"Error processing question: {error_msg}")
                else:
                    # Use simple text search and answer generation
                    docs = simple_text_search(question, text_chunks)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    answer = generate_simple_answer(question, context)
                    answers.append(f"{answer} (Note: Using fallback mode)")
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    answers.append("Rate limit exceeded. Please try again later or reduce the number of questions.")
                else:
                    answers.append(f"Error processing question: {error_msg}")

        return jsonify({"answers": answers})

    except Exception as e:
        # Generic error handler for any other issues
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
            return jsonify({"error": "API rate limit exceeded. Please try again later or reduce the number of questions."}), 429
        else:
            print(f"An unexpected error occurred: {e}")
            return jsonify({"error": "An internal server error occurred."}), 500

    finally:
        # Clean up the downloaded file
        if os.path.exists(pdf_save_path):
            os.remove(pdf_save_path)


# --- Main Execution ---

if __name__ == "__main__":
    # Ensure the Mistral API key is set
    if not os.getenv("MISTRAL_API_KEY"):
        raise ValueError("MISTRAL_API_KEY environment variable not set. Please create a .env file with your Mistral API key.")
    
    # Get port from environment variable (for cloud platforms)
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)