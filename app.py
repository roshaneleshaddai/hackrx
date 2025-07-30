import os
import requests
import asyncio
import time
import re
import warnings
from flask import Flask, request, jsonify

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Import NLTK for sentence tokenization
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Using fallback sentence tokenization.")

# Load environment variables from .env file
load_dotenv()

# Suppress warnings from Mistral tokenizer
warnings.filterwarnings("ignore", message=".*could not download mistral tokenizer.*")
warnings.filterwarnings("ignore", message=".*Falling back to a dummy tokenizer.*")

# Set HF_TOKEN if not already set to prevent tokenizer download warnings
if not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = "dummy_token"

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

def intelligent_truncate(text, max_words=80):
    """Intelligently truncate text to max_words while completing sentences."""
    words = text.split()
    
    if len(words) <= max_words:
        return text
    
    # Truncate to max_words
    truncated_words = words[:max_words]
    truncated_text = ' '.join(truncated_words)
    
    # Use NLTK for proper sentence tokenization if available
    if NLTK_AVAILABLE:
        sentences = sent_tokenize(truncated_text)
    else:
        # Fallback: simple sentence splitting
        sentences = re.split(r'[.!?]+', truncated_text)
        sentences = [s.strip() for s in sentences if s.strip()]
    
    # If we have complete sentences, return them
    if sentences:
        # Remove the last sentence if it's incomplete (doesn't end with punctuation)
        if not truncated_text.rstrip().endswith(('.', '!', '?')):
            if len(sentences) > 1:
                # Return all but the last sentence
                return ' '.join(sentences[:-1]).strip()
            else:
                # If only one sentence and it's incomplete, try to find a better break point
                return find_better_break_point(truncated_text, max_words)
        else:
            # All sentences are complete
            return truncated_text
    
    return truncated_text

def find_better_break_point(text, max_words):
    """Find a better break point in the text to avoid cutting mid-sentence."""
    words = text.split()
    
    # Look for natural break points (conjunctions, prepositions)
    break_indicators = ['and', 'or', 'but', 'however', 'therefore', 'furthermore', 'additionally', 'also', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
    
    # Start from max_words and work backwards
    for i in range(max_words - 1, max_words // 2, -1):
        if i < len(words) and words[i].lower() in break_indicators:
            # Found a good break point
            return ' '.join(words[:i]) + '.'
    
    # If no good break point found, just truncate and add ellipsis
    return ' '.join(words[:max_words-1]) + '...'

def post_process_answer(answer):
    """Clean and standardize the AI response for better consistency."""
    if not answer:
        return "No information available in the document."
    
    # Remove literal \n characters and replace with actual line breaks
    answer = answer.replace('\\n', '\n')
    
    # Remove markdown formatting
    answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer)  # Remove **bold** formatting
    answer = re.sub(r'\*([^*]+)\*', r'\1', answer)  # Remove *italic* formatting
    
    # Remove section references
    answer = re.sub(r'Section\s+\d+\.\d+\.\d+', '', answer)  # Remove "Section 3.1.15" etc.
    answer = re.sub(r'Section\s+\d+\.\d+', '', answer)  # Remove "Section 3.1" etc.
    answer = re.sub(r'Section\s+\d+', '', answer)  # Remove "Section 3" etc.
    
    # Clean up unwanted characters and formatting
    answer = re.sub(r'\\ni', '', answer)  # Remove \ni
    answer = re.sub(r'\\n\d+', '', answer)  # Remove \n followed by numbers
    answer = re.sub(r'\\n\s*[iiv]+\.', '', answer)  # Remove \n followed by roman numerals
    
    # Remove all bullet points and convert to paragraph text
    answer = re.sub(r'^\s*[-•]\s*', '', answer, flags=re.MULTILINE)  # Remove bullet points at start of lines
    answer = re.sub(r'\s*•\s*', ' ', answer)  # Remove bullet points in middle of text
    answer = re.sub(r'^\s*\d+\.\s*', '', answer, flags=re.MULTILINE)  # Remove numbered lists
    answer = re.sub(r'^\s*[iiv]+\.\s*', '', answer, flags=re.MULTILINE)  # Remove roman numerals
    
    # Remove extra whitespace and normalize line breaks
    answer = re.sub(r'\n\s*\n', ' ', answer.strip())  # Convert double line breaks to single space
    answer = re.sub(r'\n', ' ', answer)  # Convert single line breaks to spaces
    
    # Comprehensive number mapping for written numbers to digits
    number_mapping = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
        'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
        'twenty-one': '21', 'twenty-two': '22', 'twenty-three': '23', 'twenty-four': '24',
        'twenty-five': '25', 'twenty-six': '26', 'twenty-seven': '27', 'twenty-eight': '28',
        'twenty-nine': '29', 'thirty': '30', 'thirty-one': '31', 'thirty-two': '32',
        'thirty-three': '33', 'thirty-four': '34', 'thirty-five': '35', 'thirty-six': '36',
        'forty': '40', 'forty-five': '45', 'fifty': '50', 'sixty': '60', 'seventy': '70',
        'eighty': '80', 'ninety': '90', 'hundred': '100'
    }
    
    # Convert written numbers to digits
    for word, number in number_mapping.items():
        # Match whole words only to avoid partial replacements
        answer = re.sub(rf'\b{word}\b', number, answer, flags=re.IGNORECASE)
    
    # Handle special cases like "thirty-six (36)" -> "36"
    answer = re.sub(r'(\w+)\s*\((\d+)\)', r'\2', answer)
    
    # Standardize time period formatting
    answer = re.sub(r'(\d+)\s+months?\s+of\s+continuous\s+coverage', r'\1 months of continuous coverage', answer)
    answer = re.sub(r'(\d+)\s+days?', r'\1 days', answer)
    answer = re.sub(r'(\d+)\s+years?', r'\1 years', answer)
    
    # Remove any trailing "(Note: Using fallback mode)" messages
    answer = re.sub(r'\s*\(Note: Using fallback mode[^)]*\)\s*$', '', answer)
    
    # Clean up extra spaces around punctuation
    answer = re.sub(r'\s+([.,;:])', r'\1', answer)
    
    # Standardize age references (e.g., "eighteen (18) years" -> "18 years")
    answer = re.sub(r'(\w+)\s*\((\d+)\)\s*years?\s+of\s+age', r'\2 years of age', answer, flags=re.IGNORECASE)
    
    # Final cleanup of any remaining unwanted characters
    answer = re.sub(r'\\[a-zA-Z0-9]+', '', answer)  # Remove any remaining \ followed by letters/numbers
    answer = re.sub(r'\s+', ' ', answer)  # Normalize multiple spaces to single space
    
    # Clean up any remaining section references in parentheses
    answer = re.sub(r'\([^)]*Section[^)]*\)', '', answer)
    
    # Normalize phrasing like "This means" or "It implies" into policyholder-focused summaries
    answer = re.sub(r'\bThis means that\b', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'\bIt implies that\b', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'\bThis means\b', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'\bIt implies\b', '', answer, flags=re.IGNORECASE)
    
    # Start with Yes/No/Summary statement if missing
    if not answer.lower().startswith(("yes", "no", "a ", "the ", "there is", "coverage", "under", "the policy")):
        answer = "The policy states that " + answer
    
    # Intelligent truncation with sentence completion
    answer = intelligent_truncate(answer, max_words=80)
    
    # Ensure the answer is a single paragraph
    answer = re.sub(r'\s+', ' ', answer)  # Normalize all whitespace to single spaces
    answer = answer.strip()
    
    return answer

def format_structured_response(answer):
    """Format the response in a more structured and readable way."""
    if not answer:
        return "No information available in the document."
    

    # Remove any remaining markdown formatting
    answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer)  # Remove **bold** formatting
    answer = re.sub(r'\*([^*]+)\*', r'\1', answer)  # Remove *italic* formatting
    
    # Remove section references
    answer = re.sub(r'Section\s+\d+\.\d+\.\d+', '', answer)
    answer = re.sub(r'Section\s+\d+\.\d+', '', answer)
    answer = re.sub(r'Section\s+\d+', '', answer)
    
    # Remove all bullet points and convert to paragraph text
    answer = re.sub(r'^\s*[-•]\s*', '', answer, flags=re.MULTILINE)  # Remove bullet points at start of lines
    answer = re.sub(r'\s*•\s*', ' ', answer)  # Remove bullet points in middle of text
    answer = re.sub(r'^\s*\d+\.\s*', '', answer, flags=re.MULTILINE)  # Remove numbered lists
    answer = re.sub(r'^\s*[iiv]+\.\s*', '', answer, flags=re.MULTILINE)  # Remove roman numerals
    
    # Clean up any remaining section references in parentheses
    answer = re.sub(r'\([^)]*Section[^)]*\)', '', answer)
    
    # Convert line breaks to spaces for paragraph format
    answer = re.sub(r'\n', ' ', answer)
    answer = re.sub(r'\s+', ' ', answer)  # Normalize multiple spaces to single space
    
    return answer.strip()

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
        # Suppress warnings during embeddings creation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embeddings = MistralAIEmbeddings(model="mistral-embed")
            vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        return vector_store, True  # Return success flag
    except Exception as e:
        print(f"Failed to create vector store with Mistral embeddings: {e}")
        return text_chunks, False  # Return documents directly

def get_conversational_chain():
    """Creates a question-answering chain with a custom prompt and a Mistral LLM."""
    prompt_template = """
You are an expert insurance policy analyst. Answer the question strictly based on the context below.

Instructions:
- Write clearly and concisely, as if explaining to a policyholder
- Use policy terms, durations, and exact limits as provided in the context
- Include eligibility criteria, exclusions, and limits if mentioned
- Do not assume or add any extra information
- Use numerals for durations and values (e.g., "30 days", not "thirty days")
- Avoid referencing sections or using any formatting like bold or italic
- Keep the answer short, direct, and in one summarized paragraph

Response Style:
- Start with a direct answer (Yes, No, or short statement)
- Highlight key durations, conditions, or values
- Avoid technical jargon or legalese
- Do not explain common terms like "grace period" or "waiting period"
- Avoid list format; always return a single paragraph

Context:
{context}

Question:
{question}

Answer:
"""
    try:
        # Suppress warnings during model creation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ChatMistralAI(model="mistral-large-latest", temperature=0.1)
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
                        # Post-process the response for better formatting
                        processed_response = post_process_answer(response)
                        formatted_response = format_structured_response(processed_response)
                        answers.append(formatted_response)
                    except Exception as e:
                        error_msg = str(e)
                        if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                            # Fallback to simple text search
                            docs = simple_text_search(question, text_chunks)
                            context = "\n\n".join([doc.page_content for doc in docs])
                            answer = generate_simple_answer(question, context)
                            processed_answer = post_process_answer(answer)
                            formatted_answer = format_structured_response(processed_answer)
                            answers.append(f"{formatted_answer} (Note: Using fallback mode due to API limits)")
                        else:
                            answers.append(f"Error processing question: {error_msg}")
                else:
                    # Use simple text search and answer generation
                    docs = simple_text_search(question, text_chunks)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    answer = generate_simple_answer(question, context)
                    processed_answer = post_process_answer(answer)
                    formatted_answer = format_structured_response(processed_answer)
                    answers.append(f"{formatted_answer} (Note: Using fallback mode)")
                
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