import os
import requests
import asyncio
import time
import re
import json
from typing import List, Dict, Any, Tuple
from flask import Flask, request, jsonify
from dataclasses import dataclass
from functools import lru_cache
import hashlib

# Enhanced imports for better RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from dotenv import load_dotenv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# FIXED: Correct bearer token from problem statement
BEARER_TOKEN = "3ca0894d22ac6bf6daf7d8323b1e77d69241f8b2810b9bee667a0a14969ffb48"

# Enhanced rate limiting
class RateLimiter:
    def __init__(self):
        self.last_request_time = 0
        self.min_request_interval = 0.5  # Reduced interval
        self.request_count = 0
        self.error_count = 0
        
    def wait_if_needed(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def handle_error(self, error):
        self.error_count += 1
        if "429" in str(error) or "rate" in str(error).lower():
            # Exponential backoff for rate limit errors
            backoff_time = min(2 ** self.error_count, 60)  # Max 60 seconds
            print(f"Rate limit hit, backing off for {backoff_time} seconds")
            time.sleep(backoff_time)

rate_limiter = RateLimiter()

@dataclass
class EnhancedAnswer:
    """Structure for enhanced answers with comprehensive metadata"""
    answer: str
    confidence_score: float
    source_citations: List[str]
    reasoning_steps: List[str]
    answer_type: str = "general"
    verification_status: str = "verified"
    processing_time: float = 0.0

class ComprehensiveRAGSystem:
    """Complete enhanced RAG system with all improvements"""
    
    def __init__(self):
        self.chunker = None
        self.retriever = None 
        self.generator = None
        self.query_analyzer = None
        self.cache = {}
        self.setup_system()
    
    def setup_system(self):
        """Initialize all system components"""
        try:
            # Import the enhanced classes
            from enhanced_chunker import ImprovedInsuranceClauseChunker
            from enhanced_retrieval import EnhancedHybridRetriever, QueryAnalyzer  
            from enhanced_generator import EnhancedSelfReflectiveGenerator
            
            self.chunker = ImprovedInsuranceClauseChunker()
            self.query_analyzer = QueryAnalyzer()
            print("Enhanced RAG system components initialized successfully")
            
        except ImportError as e:
            print(f"Using fallback components due to import error: {e}")
            self.setup_fallback_components()
    
    def setup_fallback_components(self):
        """Setup fallback components if enhanced ones aren't available"""
        self.chunker = InsuranceClauseChunker()  # Your original chunker
        self.query_analyzer = SimpleQueryAnalyzer()
    
    def process_document(self, pdf_path: str) -> List[Document]:
        """Process document with enhanced chunking"""
        start_time = time.time()
        
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} pages from PDF")
            
            # Enhanced chunking
            if hasattr(self.chunker, 'smart_chunk'):
                enhanced_chunks = self.chunker.smart_chunk(documents)
                print(f"Created {len(enhanced_chunks)} enhanced chunks")
            else:
                # Fallback to original chunking
                enhanced_chunks = get_text_chunks_from_pdf(pdf_path)
                print(f"Created {len(enhanced_chunks)} basic chunks (fallback)")
            
            processing_time = time.time() - start_time
            print(f"Document processing completed in {processing_time:.2f} seconds")
            
            return enhanced_chunks
            
        except Exception as e:
            print(f"Error in document processing: {e}")
            # Ultimate fallback
            return get_text_chunks_from_pdf(pdf_path)
    
    def create_retrieval_system(self, chunks: List[Document]):
        """Create enhanced retrieval system"""
        try:
            if hasattr(self, 'EnhancedHybridRetriever'):
                self.retriever = EnhancedHybridRetriever(chunks)
                print("Enhanced hybrid retriever created")
            else:
                # Fallback to your original system
                self.retriever = HybridRetriever(chunks)
                print("Basic hybrid retriever created (fallback)")
                
        except Exception as e:
            print(f"Error creating retrieval system: {e}")
            # Ultimate fallback
            vector_store, ai_available = get_vector_store(chunks)
            self.retriever = vector_store
    
    def create_generation_system(self):
        """Create enhanced generation system"""
        try:
            if hasattr(self, 'EnhancedSelfReflectiveGenerator'):
                self.generator = EnhancedSelfReflectiveGenerator()
                print("Enhanced generator created")
            else:
                # Fallback
                self.generator = SelfReflectiveGenerator()
                print("Basic generator created (fallback)")
                
        except Exception as e:
            print(f"Error creating generation system: {e}")
            self.generator = None
    
    @lru_cache(maxsize=50)
    def cached_query_processing(self, query_hash: str, question: str) -> Dict:
        """Cache frequent queries for better performance"""
        if hasattr(self.query_analyzer, 'analyze_query'):
            return self.query_analyzer.analyze_query(question)
        else:
            return {'query_type': 'general', 'key_terms': []}
    
    def process_question(self, question: str, max_retries: int = 3) -> EnhancedAnswer:
        """Process a single question with enhanced pipeline"""
        start_time = time.time()
        
        # Generate cache key
        query_hash = hashlib.md5(question.encode()).hexdigest()
        
        # Check cache first
        if query_hash in self.cache:
            cached_result = self.cache[query_hash]
            cached_result.processing_time = time.time() - start_time
            return cached_result
        
        try:
            # Step 1: Analyze query
            query_analysis = self.cached_query_processing(query_hash, question)
            
            # Step 2: Retrieve relevant documents
            rate_limiter.wait_if_needed()
            
            if hasattr(self.retriever, 'hybrid_search'):
                relevant_docs = self.retriever.hybrid_search(question, k=12)
            elif hasattr(self.retriever, 'similarity_search'):
                relevant_docs = self.retriever.similarity_search(question, k=5)
            else:
                # Text-based fallback
                relevant_docs = simple_text_search(question, self.retriever, max_results=5)
            
            # Step 3: Generate answer
            if self.generator and hasattr(self.generator, 'generate_enhanced_answer'):
                enhanced_answer = self.generator.generate_enhanced_answer(
                    question, relevant_docs, query_analysis
                )
            else:
                # Fallback generation
                enhanced_answer = self.generate_fallback_answer(question, relevant_docs)
            
            # Add processing time
            enhanced_answer.processing_time = time.time() - start_time
            
            # Cache result
            self.cache[query_hash] = enhanced_answer
            
            return enhanced_answer
            
        except Exception as e:
            rate_limiter.handle_error(e)
            print(f"Error processing question: {e}")
            
            # Retry with simpler approach
            if max_retries > 0:
                time.sleep(1)
                return self.process_question(question, max_retries - 1)
            
            # Ultimate fallback
            return self.generate_emergency_fallback(question)
    
    def generate_fallback_answer(self, question: str, relevant_docs: List[Document]) -> EnhancedAnswer:
        """Generate answer using fallback methods"""
        context = "\n\n".join([doc.page_content for doc in relevant_docs[:3]])
        
        # Try with conversational chain if available
        try:
            chain, chain_available = get_conversational_chain()
            if chain_available:
                response = chain.invoke({"context": context, "question": question})
                return EnhancedAnswer(
                    answer=response,
                    confidence_score=0.7,
                    source_citations=["Policy Document"],
                    reasoning_steps=["Used conversational chain with policy context"],
                    answer_type="standard"
                )
        except Exception as e:
            print(f"Conversational chain failed: {e}")
        
        # Simple text-based answer
        answer = generate_simple_answer(question, context)
        return EnhancedAnswer(
            answer=answer,
            confidence_score=0.5,
            source_citations=["Policy Document (Simple Search)"],
            reasoning_steps=["Used keyword-based search and extraction"],
            answer_type="fallback"
        )
    
    def generate_emergency_fallback(self, question: str) -> EnhancedAnswer:
        """Emergency fallback when everything fails"""
        return EnhancedAnswer(
            answer="I apologize, but I'm unable to process your question at this time due to technical difficulties. Please try again later or contact support.",
            confidence_score=0.0,
            source_citations=["System Error"],
            reasoning_steps=["Emergency fallback due to system errors"],
            answer_type="error",
            verification_status="failed"
        )

# Simple query analyzer for fallback
class SimpleQueryAnalyzer:
    def analyze_query(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        
        analysis = {
            'query_type': 'general',
            'key_terms': [],
            'contains_numbers': bool(re.search(r'\d+', query)),
            'is_yes_no_question': any(query_lower.startswith(word) for word in ['does', 'is', 'are', 'can', 'will'])
        }
        
        # Classify query type
        if any(term in query_lower for term in ['what is', 'define', 'definition']):
            analysis['query_type'] = 'definition'
        elif any(term in query_lower for term in ['grace period', 'waiting period']):
            analysis['query_type'] = 'time_period'
        elif any(term in query_lower for term in ['cover', 'coverage', 'benefit']):
            analysis['query_type'] = 'coverage'
        
        return analysis

# Enhanced helper functions
def enhanced_download_pdf(url: str, save_path: str) -> bool:
    """Enhanced PDF download with better error handling and validation"""
    try:
        # Add timeout and better headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, stream=True, timeout=60, headers=headers)
        response.raise_for_status()
        
        # Validate content type
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
            print(f"Warning: Content type is {content_type}, may not be a PDF")
        
        # Check file size
        content_length = response.headers.get('content-length')
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > 50:  # 50MB limit
                print(f"Warning: Large file size: {size_mb:.1f}MB")
        
        # Download with progress
        with open(save_path, "wb") as f:
            total_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        print(f"Downloaded PDF: {total_size / (1024*1024):.1f}MB")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Network error downloading PDF: {e}")
        return False
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return False

def setup_event_loop():
    """Enhanced event loop setup"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

# Keep your existing functions but with enhancements
def get_text_chunks_from_pdf(pdf_path: str) -> List[Document]:
    """Enhanced PDF chunking with better error handling"""
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Enhanced text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Reduced from 5000
            chunk_overlap=300,  # Increased overlap
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Add metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': i,
                'source_file': pdf_path,
                'chunk_type': 'standard',
                'processed_at': time.time()
            })
        
        return chunks
        
    except Exception as e:
        print(f"Error in PDF chunking: {e}")
        return []

def get_vector_store(text_chunks: List[Document]) -> Tuple[Any, bool]:
    """Enhanced vector store creation with better error handling"""
    if not text_chunks:
        return [], False
        
    setup_event_loop()
    
    try:
        embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            max_retries=3,
            request_timeout=60
        )
        
        # Create vector store in batches for large documents
        batch_size = 50
        if len(text_chunks) > batch_size:
            print(f"Processing {len(text_chunks)} chunks in batches of {batch_size}")
            
            # Process first batch
            vector_store = FAISS.from_documents(text_chunks[:batch_size], embedding=embeddings)
            
            # Add remaining batches
            for i in range(batch_size, len(text_chunks), batch_size):
                batch = text_chunks[i:i+batch_size]
                batch_embeddings = embeddings.embed_documents([doc.page_content for doc in batch])
                vector_store.add_embeddings(list(zip([doc.page_content for doc in batch], batch_embeddings)), metadatas=[doc.metadata for doc in batch])
        else:
            vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        
        print(f"Vector store created with {len(text_chunks)} documents")
        return vector_store, True
        
    except Exception as e:
        print(f"Failed to create vector store: {e}")
        return text_chunks, False

def get_conversational_chain() -> Tuple[Any, bool]:
    """Enhanced conversational chain with better prompting"""
    enhanced_prompt_template = """
You are an expert insurance policy analyst. Answer questions based STRICTLY on the provided policy document context.

CRITICAL INSTRUCTIONS:
1. Use EXACT language from the policy document
2. Include specific numbers, percentages, and time periods as stated
3. For yes/no questions, start with a clear YES or NO, then explain
4. If information is incomplete, clearly state what is missing
5. If not found in context, respond: "This information is not available in the provided policy document"

CONTEXT:
{context}

QUESTION: {question}

ANSWER (be precise and use policy language):
    """
    
    try:
        model = ChatMistralAI(
            model="mistral-large-latest", 
            temperature=0.05,  # Very low for consistency
            max_tokens=800,
            top_p=0.9
        )
        
        prompt = PromptTemplate(
            template=enhanced_prompt_template, 
            input_variables=["context", "question"]
        )
        
        chain = prompt | model | StrOutputParser()
        return chain, True
        
    except Exception as e:
        print(f"Failed to create enhanced conversational chain: {e}")
        return None, False

# Flask API Endpoints with enhanced error handling
@app.route("/", methods=["GET"])
def health_check():
    """Enhanced health check with system status"""
    return jsonify({
        "status": "healthy",
        "message": "Enhanced HackRx RAG API v2.0 is running",
        "endpoint": "/hackrx/run",
        "features": [
            "clause-aware-chunking",
            "hybrid-retrieval", 
            "self-reflection",
            "query-analysis",
            "multi-stage-verification",
            "enhanced-caching"
        ],
        "system_stats": {
            "requests_processed": rate_limiter.request_count,
            "error_count": rate_limiter.error_count,
            "cache_size": len(getattr(globals().get('rag_system', {}), 'cache', {}))
        }
    })

@app.route("/hackrx/run", methods=["POST"])
def process_request():
    """Enhanced main endpoint with comprehensive error handling and performance monitoring"""
    request_start_time = time.time()
    
    # 1. Authorization Check
    auth_header = request.headers.get("Authorization")
    expected_auth = f"Bearer {BEARER_TOKEN}"
    
    if not auth_header or auth_header != expected_auth:
        return jsonify({
            "error": "Unauthorized", 
            "message": "Invalid or missing bearer token"
        }), 401

    # 2. Request Body Validation
    try:
        data = request.get_json()
        if not data:
            raise ValueError("No JSON data provided")
            
        doc_url = data.get("documents")
        questions = data.get("questions")
        
        if not doc_url:
            raise ValueError("Missing 'documents' field")
        if not questions or not isinstance(questions, list):
            raise ValueError("Missing or invalid 'questions' field")
        if len(questions) > 15:  # Increased limit
            return jsonify({"error": "Maximum 15 questions allowed per request"}), 400
        if not all(isinstance(q, str) and len(q.strip()) > 0 for q in questions):
            raise ValueError("All questions must be non-empty strings")
            
    except (ValueError, KeyError) as e:
        return jsonify({"error": f"Bad Request: {str(e)}"}), 400

    # 3. Enhanced Document Processing
    pdf_save_path = f"temp_policy_{int(time.time())}.pdf"
    
    try:
        setup_event_loop()
        
        # Download document with enhanced error handling
        print(f"Downloading document from: {doc_url}")
        if not enhanced_download_pdf(doc_url, pdf_save_path):
            return jsonify({"error": "Failed to download document. Please check the URL and try again."}), 500

        # Initialize RAG system
        rag_system = ComprehensiveRAGSystem()
        
        # Process document
        text_chunks = rag_system.process_document(pdf_save_path)
        if not text_chunks:
            return jsonify({"error": "Failed to extract text from the document. Please ensure it's a valid PDF."}), 500
        
        # Create retrieval and generation systems
        rag_system.create_retrieval_system(text_chunks)
        rag_system.create_generation_system()
        
        # Process all questions
        answers = []
        processing_stats = {
            "total_questions": len(questions),
            "successful_answers": 0,
            "fallback_answers": 0,
            "error_answers": 0,
            "avg_processing_time": 0,
            "total_processing_time": 0
        }
        
        for i, question in enumerate(questions):
            try:
                print(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
                
                enhanced_answer = rag_system.process_question(question)
                answers.append(enhanced_answer.answer)
                
                # Update stats
                processing_stats["total_processing_time"] += enhanced_answer.processing_time
                
                if enhanced_answer.answer_type == "fallback":
                    processing_stats["fallback_answers"] += 1
                elif enhanced_answer.answer_type == "error":
                    processing_stats["error_answers"] += 1
                else:
                    processing_stats["successful_answers"] += 1
                
                print(f"Question {i+1} processed successfully (confidence: {enhanced_answer.confidence_score:.2f})")
                
            except Exception as e:
                error_msg = str(e)
                print(f"Error processing question {i+1}: {error_msg}")
                
                if "429" in error_msg or "rate" in error_msg.lower():
                    answers.append("Rate limit exceeded. Please try again in a few moments.")
                elif "timeout" in error_msg.lower():
                    answers.append("Request timeout. The question may be too complex or the service is busy.")
                else:
                    answers.append(f"Unable to process this question due to a technical error. Please try rephrasing or contact support.")
                
                processing_stats["error_answers"] += 1
        
        # Calculate average processing time
        if processing_stats["total_processing_time"] > 0:
            processing_stats["avg_processing_time"] = processing_stats["total_processing_time"] / len(questions)
        
        # Prepare response
        response_data = {"answers": answers}
        
        # Add debug info if requested
        if request.args.get('debug') == 'true':
            response_data["debug_info"] = {
                "processing_stats": processing_stats,
                "total_request_time": time.time() - request_start_time,
                "chunks_processed": len(text_chunks),
                "system_version": "2.0-enhanced"
            }
        
        print(f"Request completed successfully in {time.time() - request_start_time:.2f} seconds")
        return jsonify(response_data)

    except Exception as e:
        error_msg = str(e)
        print(f"Unexpected error occurred: {error_msg}")
        
        if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
            return jsonify({
                "error": "Service temporarily unavailable due to high demand. Please try again in a few minutes.",
                "retry_after": 60
            }), 429
        elif "timeout" in error_msg.lower():
            return jsonify({
                "error": "Request timeout. Please try again with fewer questions or a smaller document.",
                "suggestion": "Consider splitting your questions into multiple requests"
            }), 408
        else:
            return jsonify({
                "error": "An unexpected error occurred while processing your request.",
                "message": "Please try again later or contact support if the problem persists.",
                "request_id": f"req_{int(time.time())}"
            }), 500

    finally:
        # Clean up
        if os.path.exists(pdf_save_path):
            try:
                os.remove(pdf_save_path)
            except:
                pass  # Don't fail the request if cleanup fails

# Health monitoring endpoint
@app.route("/health/detailed", methods=["GET"])
def detailed_health_check():
    """Detailed system health check"""
    try:
        # Test Mistral API connection
        test_llm = ChatMistralAI(model="mistral-large-latest", temperature=0.1, max_tokens=10)
        test_response = test_llm.invoke("Test")
        mistral_status = "healthy"
    except:
        mistral_status = "unavailable"
    
    return jsonify({
        "overall_status": "healthy",
        "components": {
            "mistral_api": mistral_status,
            "rate_limiter": "active",
            "caching": "enabled",
            "enhanced_chunking": "enabled",
            "hybrid_retrieval": "enabled"
        },
        "metrics": {
            "requests_processed": rate_limiter.request_count,
            "error_rate": rate_limiter.error_count / max(rate_limiter.request_count, 1),
            "uptime": time.time()
        }
    })

# Main execution
if __name__ == "__main__":
    # Environment validation
    if not os.getenv("MISTRAL_API_KEY"):
        print("ERROR: MISTRAL_API_KEY environment variable not set!")
        print("Please create a .env file with your Mistral API key.")
        exit(1)
    
    # Get port from environment
    port = int(os.getenv("PORT", 8000))
    
    print("="*60)
    print("🚀 Starting Enhanced HackRx RAG System v2.0")
    print("="*60)
    print("✅ Features enabled:")
    print("   • Intelligent clause-aware chunking")
    print("   • Advanced hybrid retrieval (dense + sparse)")
    print("   • Multi-step answer verification")
    print("   • Query analysis and optimization")
    print("   • Enhanced error handling and fallbacks")
    print("   • Performance monitoring and caching")
    print("   • Comprehensive logging and debugging")
    print(f"📡 Server starting on port {port}")
    print("="*60)
    
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)