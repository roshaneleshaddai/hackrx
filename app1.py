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

class RateLimiter:
    """Enhanced rate limiting with exponential backoff"""
    def __init__(self):
        self.last_request_time = 0
        self.min_request_interval = 0.5
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
            backoff_time = min(2 ** self.error_count, 60)
            print(f"Rate limit hit, backing off for {backoff_time} seconds")
            time.sleep(backoff_time)

rate_limiter = RateLimiter()

class ImprovedInsuranceClauseChunker:
    """Enhanced chunker specifically designed for insurance policies"""
    
    def __init__(self):
        self.clause_patterns = [
            r'^\d+\.\s+[A-Z][A-Z\s]+:?$',  # "3. DEFINITIONS:"
            r'^\d+\.\d+\s+[A-Z]',          # "3.1 Accident means"
            r'^[A-Z][a-z]+\s+means\s',     # "Grace Period means"
            r'^EXCLUSIONS?:?\s*$',         # Exclusion headers
            r'^WAITING PERIOD:?\s*$',      # Waiting period sections
            r'^COVERAGE:?\s*$',            # Coverage sections
            r'^CONDITIONS:?\s*$',          # Conditions sections
            r'^DEFINITIONS?:?\s*$',        # Definitions sections
            r'^\([a-z]\)\s+',              # "(a) condition text"
            r'^\([ivx]+\)\s+',             # "(i) roman numeral items"
        ]
        
        self.preserve_together = [
            'waiting period', 'grace period', 'sum insured', 'co-payment',
            'pre-existing condition', 'no claim discount', 'room rent',
            'icu charges', 'deductible', 'sub-limit', 'maternity benefit'
        ]
    
    def smart_chunk(self, documents: List[Document]) -> List[Document]:
        """Enhanced chunking with better boundary detection"""
        enhanced_chunks = []
        
        for doc in documents:
            text = doc.page_content
            sections = self._identify_major_sections(text)
            
            for section_title, section_text in sections:
                chunks = self._intelligent_section_chunking(section_text, section_title)
                
                for i, chunk_text in enumerate(chunks):
                    enhanced_chunk = Document(
                        page_content=chunk_text,
                        metadata={
                            **doc.metadata,
                            'section': section_title,
                            'chunk_index': i,
                            'total_chunks_in_section': len(chunks),
                            'chunk_type': self._classify_chunk_type(chunk_text),
                            'contains_numbers': self._contains_important_numbers(chunk_text),
                            'contains_conditions': self._contains_conditions(chunk_text)
                        }
                    )
                    enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def _identify_major_sections(self, text: str) -> List[Tuple[str, str]]:
        """Better section identification with context preservation"""
        sections = []
        lines = text.split('\n')
        current_section = "GENERAL"
        current_text = ""
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            is_section_header = False
            for pattern in self.clause_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    is_section_header = True
                    break
            
            insurance_headers = [
                'DEFINITIONS', 'EXCLUSIONS', 'WAITING PERIOD', 'COVERAGE',
                'CONDITIONS', 'BENEFITS', 'LIMITATIONS', 'CLAIMS PROCEDURE'
            ]
            
            if any(header in line_stripped.upper() for header in insurance_headers):
                is_section_header = True
            
            if is_section_header and current_text.strip():
                sections.append((current_section, current_text.strip()))
                current_section = line_stripped
                current_text = ""
                
                context_lines = 2
                start_idx = max(0, i - context_lines)
                context = '\n'.join(lines[start_idx:i])
                if context.strip():
                    current_text = f"Previous context:\n{context}\n\n"
            
            current_text += line + "\n"
        
        if current_text.strip():
            sections.append((current_section, current_text.strip()))
        
        return sections
    
    def _intelligent_section_chunking(self, text: str, section_title: str) -> List[str]:
        """Intelligent chunking based on section type"""
        section_upper = section_title.upper()
        
        if 'DEFINITION' in section_upper:
            return self._chunk_definitions_improved(text)
        elif any(keyword in section_upper for keyword in ['EXCLUSION', 'COVERAGE', 'BENEFIT']):
            return self._chunk_clauses_improved(text)
        elif 'WAITING' in section_upper or 'GRACE' in section_upper:
            return self._chunk_time_periods(text)
        else:
            return self._chunk_general_improved(text)
    
    def _chunk_definitions_improved(self, text: str) -> List[str]:
        """Enhanced definition chunking"""
        chunks = []
        current_chunk = ""
        
        definition_splits = re.split(r'(\d+\.\d+\.?\s+[A-Z][^:]*(?:means|shall mean|is defined as))', 
                                   text, flags=re.IGNORECASE)
        
        for i, part in enumerate(definition_splits):
            if not part.strip():
                continue
                
            if re.match(r'\d+\.\d+\.?\s+[A-Z][^:]*(?:means|shall mean|is defined as)', 
                       part, re.IGNORECASE):
                if current_chunk.strip() and len(current_chunk) > 100:
                    chunks.append(current_chunk.strip())
                current_chunk = part
            else:
                current_chunk += part
                
            if len(current_chunk) > 3000:
                break_point = self._find_sentence_break(current_chunk, 2500)
                if break_point > 0:
                    chunks.append(current_chunk[:break_point].strip())
                    current_chunk = current_chunk[break_point:]
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_clauses_improved(self, text: str) -> List[str]:
        """Enhanced clause chunking with better boundary detection"""
        chunks = []
        
        clause_markers = [
            r'(\([a-z]\)\s+)',
            r'(\([ivx]+\)\s+)',
            r'(\d+\.\d+\.?\s+)',
            r'((?:The Company|The Insurer|We|This Policy)\s+(?:shall|will|does|covers?))'
        ]
        
        combined_pattern = '|'.join(clause_markers)
        parts = re.split(combined_pattern, text, flags=re.IGNORECASE)
        
        current_chunk = ""
        for part in parts:
            if not part or part.isspace():
                continue
                
            if len(current_chunk) + len(part) > 2500 and current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = part
            else:
                current_chunk += part
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_time_periods(self, text: str) -> List[str]:
        """Special chunking for time-sensitive information"""
        time_patterns = [
            r'\d+\s*(?:days?|months?|years?)',
            r'(?:thirty|sixty|ninety)\s*(?:days?|months?)',
            r'grace period', r'waiting period'
        ]
        
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            has_time_info = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in time_patterns)
            
            if has_time_info and current_chunk and len(current_chunk) > 1000:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                
            if len(current_chunk) > 3000:
                chunks.append(current_chunk.strip())
                current_chunk = ""
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_general_improved(self, text: str) -> List[str]:
        """Improved general chunking with semantic boundaries"""
        chunks = []
        current_chunk = ""
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > 2000 and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _find_sentence_break(self, text: str, target_pos: int) -> int:
        """Find the best sentence break near target position"""
        for i in range(target_pos, min(len(text), target_pos + 200)):
            if text[i] in '.!?':
                if i + 1 < len(text) and text[i + 1].isspace():
                    return i + 1
        
        for i in range(target_pos, max(0, target_pos - 200), -1):
            if text[i] in '.!?':
                if i + 1 < len(text) and text[i + 1].isspace():
                    return i + 1
        
        return 0
    
    def _classify_chunk_type(self, text: str) -> str:
        """Classify the type of chunk for better retrieval"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['means', 'defined as', 'definition']):
            return 'definition'
        elif any(term in text_lower for term in ['exclusion', 'excluded', 'not covered']):
            return 'exclusion'
        elif any(term in text_lower for term in ['waiting period', 'wait for']):
            return 'waiting_period'
        elif any(term in text_lower for term in ['grace period']):
            return 'grace_period'
        elif any(term in text_lower for term in ['coverage', 'covers', 'benefit']):
            return 'coverage'
        elif any(term in text_lower for term in ['condition', 'provided that', 'subject to']):
            return 'condition'
        else:
            return 'general'
    
    def _contains_important_numbers(self, text: str) -> bool:
        """Check if chunk contains important numerical information"""
        number_patterns = [
            r'\d+%',
            r'\d+\s*(?:days?|months?|years?)',
            r'Rs\.?\s*\d+',
            r'\d+\s*(?:lakh|crore)',
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in number_patterns)
    
    def _contains_conditions(self, text: str) -> bool:
        """Check if chunk contains conditional statements"""
        condition_indicators = [
            'provided that', 'subject to', 'if', 'unless', 'except',
            'conditions', 'requirements', 'eligibility'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in condition_indicators)

class QueryAnalyzer:
    """Analyze queries to understand intent and type"""
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to understand intent and requirements"""
        analysis = {
            'query_type': self._classify_query_type(query),
            'contains_numbers': self._contains_numbers(query),
            'is_yes_no_question': self._is_yes_no_question(query),
            'key_terms': self._extract_key_terms(query),
            'time_indicators': self._extract_time_indicators(query),
            'medical_terms': self._extract_medical_terms(query)
        }
        
        return analysis
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['what is', 'define', 'definition', 'means']):
            return 'definition'
        elif any(term in query_lower for term in ['grace period', 'waiting period', 'days', 'months']):
            return 'time_period'
        elif any(term in query_lower for term in ['cover', 'coverage', 'benefit', 'include']):
            return 'coverage'
        elif any(term in query_lower for term in ['exclusion', 'excluded', 'not covered']):
            return 'exclusion'
        elif any(term in query_lower for term in ['how much', 'amount', 'cost', 'premium']):
            return 'financial'
        elif re.search(r'\d+', query):
            return 'specific_numbers'
        else:
            return 'general'
    
    def _contains_numbers(self, query: str) -> bool:
        """Check if query contains specific numbers"""
        return bool(re.search(r'\d+', query))
    
    def _is_yes_no_question(self, query: str) -> bool:
        """Check if query expects yes/no answer"""
        query_lower = query.lower()
        yes_no_indicators = [
            'does', 'is', 'are', 'can', 'will', 'would', 'should',
            'do you', 'is there', 'are there', 'does this'
        ]
        return any(query_lower.startswith(indicator) for indicator in yes_no_indicators)
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key insurance terms from query"""
        insurance_terms = [
            'grace period', 'waiting period', 'sum insured', 'co-payment',
            'pre-existing', 'maternity', 'surgery', 'hospital', 'premium',
            'coverage', 'benefit', 'exclusion', 'deductible', 'claim'
        ]
        
        query_lower = query.lower()
        found_terms = []
        
        for term in insurance_terms:
            if term in query_lower:
                found_terms.append(term)
        
        return found_terms
    
    def _extract_time_indicators(self, query: str) -> List[str]:
        """Extract time-related terms from query"""
        time_patterns = [
            r'\d+\s*(?:days?|months?|years?)',
            r'(?:thirty|sixty|ninety|one|two|three)\s*(?:days?|months?|years?)',
            r'grace period', r'waiting period'
        ]
        
        time_indicators = []
        for pattern in time_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            time_indicators.extend(matches)
        
        return time_indicators
    
    def _extract_medical_terms(self, query: str) -> List[str]:
        """Extract medical terms from query"""
        medical_terms = [
            'surgery', 'operation', 'treatment', 'procedure', 'diagnosis',
            'cataract', 'maternity', 'pregnancy', 'delivery', 'consultation',
            'hospitalization', 'icu', 'emergency'
        ]
        
        query_lower = query.lower()
        found_terms = []
        
        for term in medical_terms:
            if term in query_lower:
                found_terms.append(term)
        
        return found_terms

class EnhancedHybridRetriever:
    """Advanced retrieval with query understanding and multi-stage filtering"""
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.vector_store = None
        self.tfidf_vectorizer = None
        self.doc_tfidf_matrix = None
        self.setup_retrievers()
    
    def setup_retrievers(self):
        """Enhanced retriever setup with better preprocessing"""
        try:
            embeddings = MistralAIEmbeddings(model="mistral-embed")
            self.vector_store = FAISS.from_documents(self.documents, embedding=embeddings)
            
            doc_texts = [self._preprocess_for_tfidf(doc.page_content) for doc in self.documents]
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=10000,
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True,
                analyzer='word',
                token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
            )
            self.doc_tfidf_matrix = self.tfidf_vectorizer.fit_transform(doc_texts)
            
        except Exception as e:
            print(f"Error setting up enhanced retrievers: {e}")
            self.vector_store = None
    
    def _preprocess_for_tfidf(self, text: str) -> str:
        """Preprocess text for better TF-IDF performance"""
        text = re.sub(r'pre-existing', 'preexisting', text, flags=re.IGNORECASE)
        text = re.sub(r'co-payment', 'copayment', text, flags=re.IGNORECASE)
        text = re.sub(r'sub-limit', 'sublimit', text, flags=re.IGNORECASE)
        
        text = re.sub(r'(\d+)\s*days?', r'\1days', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*months?', r'\1months', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*years?', r'\1years', text, flags=re.IGNORECASE)
        
        return text
    
    def hybrid_search(self, query: str, k: int = 15) -> List[Document]:
        """Enhanced hybrid search with query analysis"""
        dense_results = self._enhanced_dense_search(query, k)
        sparse_results = self._enhanced_sparse_search(query, k)
        
        combined_results = self._intelligent_combine(dense_results, sparse_results)
        reranked_results = self._multi_stage_rerank(query, combined_results, k//2)
        
        return reranked_results
    
    def _enhanced_dense_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Enhanced dense search with query expansion"""
        if self.vector_store is None:
            return []
        
        try:
            original_results = self.vector_store.similarity_search_with_score(query, k=k//2)
            expanded_query = self._expand_query(query)
            expanded_results = self.vector_store.similarity_search_with_score(expanded_query, k=k//2)
            
            all_results = original_results + expanded_results
            seen_content = set()
            unique_results = []
            
            for doc, score in all_results:
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_results.append((doc, score))
            
            return unique_results[:k]
            
        except Exception as e:
            print(f"Enhanced dense search error: {e}")
            return []
    
    def _enhanced_sparse_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Enhanced sparse search with term boosting"""
        if self.tfidf_vectorizer is None or self.doc_tfidf_matrix is None:
            return []
        
        try:
            processed_query = self._preprocess_for_tfidf(query)
            boosted_query = self._create_boosted_query(processed_query, query)
            
            query_tfidf = self.tfidf_vectorizer.transform([boosted_query])
            similarities = cosine_similarity(query_tfidf, self.doc_tfidf_matrix).flatten()
            
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.05:
                    results.append((self.documents[idx], similarities[idx]))
            
            return results
            
        except Exception as e:
            print(f"Enhanced sparse search error: {e}")
            return []
    
    def _expand_query(self, query: str) -> str:
        """Expand query with domain-specific synonyms"""
        synonyms = {
            'grace period': ['grace period', 'premium payment grace', 'payment grace period'],
            'waiting period': ['waiting period', 'wait period', 'waiting time', 'moratorium period'],
            'maternity': ['maternity', 'pregnancy', 'childbirth', 'delivery'],
            'pre-existing': ['pre-existing', 'preexisting', 'pre existing', 'prior condition'],
            'coverage': ['coverage', 'cover', 'benefit', 'protection'],
            'exclusion': ['exclusion', 'excluded', 'not covered', 'limitation'],
        }
        
        expanded_query = query
        query_lower = query.lower()
        
        for term, alternatives in synonyms.items():
            if term in query_lower:
                expanded_query += " " + " ".join(alternatives)
        
        return expanded_query
    
    def _create_boosted_query(self, query: str, original_query: str) -> str:
        """Create boosted query for TF-IDF search"""
        boosted_query = query
        
        if any(term in original_query.lower() for term in ['grace period', 'waiting period']):
            time_terms = ['days', 'months', 'years', 'period', 'waiting', 'grace']
            for term in time_terms:
                if term in query.lower():
                    boosted_query += f" {term} {term} {term}"
        
        return boosted_query
    
    def _intelligent_combine(self, dense_results: List[Tuple[Document, float]], 
                           sparse_results: List[Tuple[Document, float]]) -> List[Document]:
        """Intelligently combine results"""
        dense_weight = 0.6
        sparse_weight = 0.4
        
        doc_scores = {}
        
        for doc, score in dense_results:
            doc_key = hash(doc.page_content[:200])
            doc_scores[doc_key] = (dense_weight * score, doc)
        
        for doc, score in sparse_results:
            doc_key = hash(doc.page_content[:200])
            if doc_key in doc_scores:
                doc_scores[doc_key] = (doc_scores[doc_key][0] + sparse_weight * score, doc)
            else:
                doc_scores[doc_key] = (sparse_weight * score, doc)
        
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in sorted_docs]
    
    def _multi_stage_rerank(self, query: str, results: List[Document], k: int) -> List[Document]:
        """Multi-stage reranking for better accuracy"""
        if not results:
            return results
        
        scored_results = []
        query_words = set(query.lower().split())
        
        for doc in results:
            score = self._calculate_relevance_score(query, doc, query_words)
            scored_results.append((score, doc))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_results[:k]]
    
    def _calculate_relevance_score(self, query: str, doc: Document, query_words: set) -> float:
        """Calculate relevance score between query and document"""
        score = 0.0
        doc_lower = doc.page_content.lower()
        
        # Exact phrase matching
        if query.lower() in doc_lower:
            score += 2.0
        
        # Keyword overlap
        doc_words = set(re.findall(r'\b\w+\b', doc_lower))
        overlap = len(query_words.intersection(doc_words)) / len(query_words) if query_words else 0
        score += overlap * 1.5
        
        # Metadata-based boosting
        doc_type = doc.metadata.get('chunk_type', '')
        if 'definition' in query.lower() and doc_type == 'definition':
            score += 1.0
        elif any(term in query.lower() for term in ['grace period', 'waiting period']) and 'period' in doc_type:
            score += 1.0
        elif 'coverage' in query.lower() and doc_type == 'coverage':
            score += 1.0
        
        # Early appearance bonus
        if any(word in doc_lower[:500] for word in query_words):
            score += 0.5
        
        return score

class EnhancedSelfReflectiveGenerator:
    """Advanced generator with multi-step verification and answer refinement"""
    
    def __init__(self):
        try:
            self.llm = ChatMistralAI(
                model="mistral-large-latest", 
                temperature=0.05,  # Very low temperature for consistency
                max_tokens=1000,
                top_p=0.9
            )
            self.available = True
        except Exception as e:
            print(f"LLM initialization failed: {e}")
            self.available = False
    
    def generate_enhanced_answer(self, question: str, context_docs: List[Document], 
                               query_analysis: Dict = None) -> EnhancedAnswer:
        """Generate answer with comprehensive verification pipeline"""
        if not self.available:
            return self._fallback_answer(question, context_docs)
        
        try:
            prepared_context = self._prepare_context(context_docs, question, query_analysis)
            initial_answer = self._generate_initial_answer_v2(question, prepared_context, query_analysis)
            
            verification_results = self._comprehensive_verification(question, initial_answer, prepared_context)
            
            if verification_results['needs_refinement']:
                refined_answer = self._refine_answer(question, initial_answer, prepared_context, verification_results)
            else:
                refined_answer = initial_answer
            
            final_answer = self._final_quality_check(question, refined_answer, prepared_context)
            
            confidence_score = self._calculate_enhanced_confidence(question, final_answer, context_docs, verification_results)
            source_citations = self._generate_precise_citations(context_docs, final_answer)
            reasoning_steps = self._extract_detailed_reasoning(question, final_answer, verification_results)
            
            return EnhancedAnswer(
                answer=final_answer,
                confidence_score=confidence_score,
                source_citations=source_citations,
                reasoning_steps=reasoning_steps
            )
            
        except Exception as e:
            print(f"Enhanced generation failed: {e}")
            return self._fallback_answer(question, context_docs)
    
    def _prepare_context(self, context_docs: List[Document], question: str, 
                        query_analysis: Dict = None) -> str:
        """Prepare and optimize context for better answer generation"""
        if not context_docs:
            return ""
        
        scored_docs = []
        question_words = set(question.lower().split())
        
        for doc in context_docs:
            doc_words = set(doc.page_content.lower().split())
            overlap_score = len(question_words.intersection(doc_words)) / len(question_words)
            
            doc_type = doc.metadata.get('chunk_type', '')
            type_boost = 0
            if query_analysis:
                if query_analysis.get('query_type') == 'definition' and doc_type == 'definition':
                    type_boost = 0.3
                elif query_analysis.get('query_type') == 'time_period' and 'period' in doc_type:
                    type_boost = 0.3
                elif query_analysis.get('query_type') == 'coverage' and doc_type == 'coverage':
                    type_boost = 0.3
            
            total_score = overlap_score + type_boost
            scored_docs.append((total_score, doc))
        
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        prepared_context = ""
        for i, (score, doc) in enumerate(scored_docs[:5]):
            section = doc.metadata.get('section', f'Section {i+1}')
            chunk_type = doc.metadata.get('chunk_type', 'general')
            
            prepared_context += f"\n--- DOCUMENT {i+1} (Type: {chunk_type}, Section: {section}) ---\n"
            prepared_context += doc.page_content
            prepared_context += f"\n--- END DOCUMENT {i+1} ---\n"
        
        return prepared_context
    
    def _generate_initial_answer_v2(self, question: str, context: str, 
                                  query_analysis: Dict = None) -> str:
        """Generate initial answer with enhanced, context-aware prompting"""
        
        answer_format = "comprehensive"
        if query_analysis:
            if query_analysis.get('is_yes_no_question'):
                answer_format = "yes_no_with_explanation"
            elif query_analysis.get('query_type') == 'definition':
                answer_format = "definition"
            elif query_analysis.get('query_type') == 'time_period':
                answer_format = "time_specific"
        
        enhanced_prompt = self._create_specialized_prompt(question, context, answer_format, query_analysis)
        
        try:
            response = self.llm.invoke(enhanced_prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Initial answer generation failed: {e}")
            return self._extract_answer_from_context(question, context)
    
    def _create_specialized_prompt(self, question: str, context: str, 
                                 answer_format: str, query_analysis: Dict = None) -> str:
        """Create specialized prompts based on question type"""
        
        base_instructions = """
You are an expert insurance policy analyst. Your task is to provide accurate, precise answers based STRICTLY on the provided policy documents.

CRITICAL REQUIREMENTS:
1. Use EXACT terminology from the policy documents
2. Include specific numbers, percentages, time periods EXACTLY as stated
3. If multiple conditions apply, list them clearly and completely
4. Quote relevant sections when providing specific details
5. If information is not in the provided context, state: "This information is not available in the provided policy documents"
6. Never make assumptions or add information not present in the documents
"""
        
        if answer_format == "yes_no_with_explanation":
            format_instruction = """
ANSWER FORMAT REQUIRED:
- Start with a clear YES or NO
- Follow with a detailed explanation including:
  * Specific conditions that apply
  * Any limitations or exclusions
  * Relevant time periods or amounts
  * Exact policy language where applicable
"""
        
        elif answer_format == "definition":
            format_instruction = """
ANSWER FORMAT REQUIRED:
- Provide the exact definition from the policy
- Include any sub-definitions or clarifications
- Mention the specific section/clause where defined
- If there are multiple related definitions, include them all
"""
        
        elif answer_format == "time_specific":
            format_instruction = """
ANSWER FORMAT REQUIRED:
- State the specific time period clearly
- Include any conditions that affect the time period
- Mention if different time periods apply to different situations
- Provide exact policy language regarding timing
"""
        
        else:
            format_instruction = """
ANSWER FORMAT REQUIRED:
- Provide a comprehensive answer covering all aspects of the question
- Structure your response with clear points if multiple aspects are involved
- Include specific details like amounts, percentages, time periods
- Mention any important conditions or limitations
"""
        
        specialized_prompt = f"""{base_instructions}

{format_instruction}

POLICY CONTEXT:
{context}

QUESTION: {question}

ANSWER (follow the format requirements exactly):"""
        
        return specialized_prompt
    
    def _comprehensive_verification(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """Multi-stage comprehensive verification of the generated answer"""
        verification_results = {
            'needs_refinement': False,
            'factual_errors': [],
            'missing_information': [],
            'formatting_issues': [],
            'confidence_factors': {}
        }
        
        factual_check = self._verify_factual_accuracy(answer, context)
        verification_results['factual_errors'] = factual_check['errors']
        verification_results['confidence_factors']['factual_accuracy'] = factual_check['accuracy_score']
        
        completeness_check = self._verify_completeness(question, answer, context)
        verification_results['missing_information'] = completeness_check['missing']
        verification_results['confidence_factors']['completeness'] = completeness_check['completeness_score']
        
        format_check = self._verify_format_and_clarity(question, answer)
        verification_results['formatting_issues'] = format_check['issues']
        verification_results['confidence_factors']['clarity'] = format_check['clarity_score']
        
        terminology_check = self._verify_insurance_terminology(answer, context)
        verification_results['confidence_factors']['terminology'] = terminology_check['terminology_score']
        
        if (len(verification_results['factual_errors']) > 0 or 
            len(verification_results['missing_information']) > 0 or
            len(verification_results['formatting_issues']) > 0 or
            verification_results['confidence_factors']['factual_accuracy'] < 0.8):
            verification_results['needs_refinement'] = True
        
        return verification_results
    
    def _verify_factual_accuracy(self, answer: str, context: str) -> Dict[str, Any]:
        """Verify factual accuracy of the answer against context"""
        verification_prompt = f"""
Compare the ANSWER against the POLICY CONTEXT and identify any factual errors.

POLICY CONTEXT:
{context}

ANSWER TO VERIFY:
{answer}

Check for:
1. Incorrect numbers, percentages, or time periods
2. Misstatement of conditions or requirements
3. Incorrect terminology or definitions
4. Claims not supported by the policy text

Respond in JSON format:
{{
    "errors": ["list specific factual errors found"],
    "accuracy_score": 0.0-1.0,
    "supporting_evidence": ["quotes from policy that support correct facts"]
}}
"""
        
        try:
            response = self.llm.invoke(verification_prompt)
            return json.loads(response.content)
        except:
            return self._simple_factual_check(answer, context)
    
    def _verify_completeness(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """Verify if the answer completely addresses the question"""
        completeness_prompt = f"""
Evaluate if the ANSWER completely addresses all parts of the QUESTION based on available information in the CONTEXT.

QUESTION: {question}
ANSWER: {answer}
CONTEXT: {context}

Check if the answer addresses:
1. All parts of the question (if multi-part)
2. Relevant conditions and limitations
3. Specific details requested (amounts, time periods, etc.)
4. Important related information available in context

Respond in JSON format:
{{
    "missing": ["list what important information is missing from the answer"],
    "completeness_score": 0.0-1.0,
    "suggestions": ["suggestions for completing the answer"]
}}
"""
        
        try:
            response = self.llm.invoke(completeness_prompt)
            return json.loads(response.content)
        except:
            return {"missing": [], "completeness_score": 0.7, "suggestions": []}
    
    def _verify_format_and_clarity(self, question: str, answer: str) -> Dict[str, Any]:
        """Verify answer format and clarity"""
        issues = []
        clarity_score = 1.0
        
        if len(answer.strip()) < 20:
            issues.append("Answer is too short")
            clarity_score -= 0.3
        
        if "This information is not available" in answer and len(answer) < 100:
            clarity_score = max(clarity_score, 0.8)
        
        if any(phrase in answer.lower() for phrase in ['maybe', 'possibly', 'might be', 'unclear']):
            issues.append("Answer contains uncertain language")
            clarity_score -= 0.2
        
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        if len(question_words.intersection(answer_words)) < len(question_words) * 0.3:
            issues.append("Answer may not directly address the question")
            clarity_score -= 0.2
        
        return {
            'issues': issues,
            'clarity_score': max(0.0, clarity_score)
        }
    
    def _verify_insurance_terminology(self, answer: str, context: str) -> Dict[str, Any]:
        """Verify proper use of insurance terminology"""
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        terminology_score = 0.5
        
        insurance_terms = [
            'grace period', 'waiting period', 'sum insured', 'deductible',
            'co-payment', 'pre-existing', 'exclusion', 'benefit', 'coverage',
            'premium', 'policy', 'claim', 'insured', 'insurer'
        ]
        
        terms_used_correctly = 0
        total_terms_in_context = 0
        
        for term in insurance_terms:
            if term in context_lower:
                total_terms_in_context += 1
                if term in answer_lower:
                    terms_used_correctly += 1
        
        if total_terms_in_context > 0:
            terminology_score += 0.4 * (terms_used_correctly / total_terms_in_context)
        
        return {'terminology_score': min(1.0, terminology_score)}
    
    def _simple_factual_check(self, answer: str, context: str) -> Dict[str, Any]:
        """Simple fallback factual check"""
        errors = []
        accuracy_score = 0.8
        
        answer_numbers = re.findall(r'\d+', answer)
        context_numbers = re.findall(r'\d+', context)
        
        for num in answer_numbers:
            if num not in context_numbers:
                errors.append(f"Number '{num}' in answer not found in policy context")
                accuracy_score -= 0.2
        
        return {
            'errors': errors,
            'accuracy_score': max(0.0, accuracy_score),
            'supporting_evidence': []
        }
    
    def _refine_answer(self, question: str, initial_answer: str, context: str, 
                      verification_results: Dict) -> str:
        """Refine the answer based on verification results"""
        refinement_prompt = f"""
The initial answer has the following issues that need to be addressed:

FACTUAL ERRORS: {verification_results.get('factual_errors', [])}
MISSING INFORMATION: {verification_results.get('missing_information', [])}
FORMATTING ISSUES: {verification_results.get('formatting_issues', [])}

ORIGINAL QUESTION: {question}
INITIAL ANSWER: {initial_answer}
POLICY CONTEXT: {context}

Please provide a refined answer that:
1. Corrects any factual errors using exact information from the policy context
2. Includes any missing important information
3. Improves clarity and formatting
4. Maintains strict adherence to the policy language

REFINED ANSWER:"""
        
        try:
            response = self.llm.invoke(refinement_prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Answer refinement failed: {e}")
            return initial_answer
    
    def _final_quality_check(self, question: str, answer: str, context: str) -> str:
        """Final quality check and minor formatting improvements"""
        answer = re.sub(r'\b(the policy states that|according to the policy|as per the policy)\b', '', answer, flags=re.IGNORECASE)
        
        insurance_terms = {
            'grace period': 'Grace Period',
            'waiting period': 'Waiting Period',
            'sum insured': 'Sum Insured',
            'no claim discount': 'No Claim Discount'
        }
        
        for term, proper_term in insurance_terms.items():
            answer = re.sub(r'\b' + re.escape(term) + r'\b', proper_term, answer, flags=re.IGNORECASE)
        
        return answer.strip()
    
    def _calculate_enhanced_confidence(self, question: str, answer: str, 
                                     context_docs: List[Document], 
                                     verification_results: Dict) -> float:
        """Calculate enhanced confidence score"""
        base_confidence = 0.3
        
        if verification_results:
            factual_accuracy = verification_results.get('confidence_factors', {}).get('factual_accuracy', 0.5)
            completeness = verification_results.get('confidence_factors', {}).get('completeness', 0.5)
            clarity = verification_results.get('confidence_factors', {}).get('clarity', 0.5)
            terminology = verification_results.get('confidence_factors', {}).get('terminology', 0.5)
            
            verification_confidence = (factual_accuracy + completeness + clarity + terminology) / 4
            base_confidence += verification_confidence * 0.4
        
        if re.search(r'\d+\s*(?:%|days?|months?|years?|rupees?|rs\.?)', answer, re.IGNORECASE):
            base_confidence += 0.15
        
        if any(term in answer.lower() for term in ['grace period', 'waiting period', 'coverage', 'exclusion']):
            base_confidence += 0.1
        
        if len(context_docs) >= 3:
            base_confidence += 0.1
        
        if 50 <= len(answer) <= 500:
            base_confidence += 0.05
        
        return min(1.0, base_confidence)
    
    def _generate_precise_citations(self, context_docs: List[Document], answer: str) -> List[str]:
        """Generate precise citations based on answer content"""
        citations = []
        answer_lower = answer.lower()
        
        for i, doc in enumerate(context_docs[:3]):
            doc_lower = doc.page_content.lower()
            
            answer_words = set(re.findall(r'\b\w+\b', answer_lower))
            doc_words = set(re.findall(r'\b\w+\b', doc_lower))
            overlap = len(answer_words.intersection(doc_words)) / len(answer_words) if answer_words else 0
            
            if overlap > 0.3:
                section = doc.metadata.get('section', f'Section {i+1}')
                chunk_type = doc.metadata.get('chunk_type', 'general')
                citations.append(f"Policy Reference: {section} ({chunk_type})")
        
        return citations if citations else ["Policy Document"]
    
    def _extract_detailed_reasoning(self, question: str, answer: str, 
                                  verification_results: Dict) -> List[str]:
        """Extract detailed reasoning steps"""
        steps = [
            f"1. Analyzed question type and requirements",
            f"2. Retrieved relevant policy sections",
            f"3. Extracted specific terms and conditions",
            f"4. Verified factual accuracy against policy text"
        ]
        
        if verification_results.get('needs_refinement'):
            steps.append("5. Refined answer based on verification results")
        
        steps.append("6. Formatted final response with precise policy language")
        
        return steps
    
    def _extract_answer_from_context(self, question: str, context: str) -> str:
        """Fallback method to extract answer from context when LLM fails"""
        if not context:
            return "The answer is not available in the provided policy documents."
        
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        sentences = re.split(r'[.!?]+', context)
        
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            if len(sentence.strip()) < 20:
                continue
                
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            score = len(question_words.intersection(sentence_words)) / len(question_words) if question_words else 0
            
            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()
        
        return best_sentence if best_sentence else "The answer is not available in the provided policy documents."
    
    def _fallback_answer(self, question: str, context_docs: List[Document]) -> EnhancedAnswer:
        """Enhanced fallback when LLM is not available"""
        context = "\n\n".join([doc.page_content for doc in context_docs])
        answer = self._extract_answer_from_context(question, context)
        
        return EnhancedAnswer(
            answer=f"{answer} (Note: Generated using fallback mode due to API unavailability)",
            confidence_score=0.4,
            source_citations=["Policy Document (Fallback Mode)"],
            reasoning_steps=["Used keyword-based search and extraction"]
        )

class ComprehensiveRAGSystem:
    """Complete enhanced RAG system with all improvements"""
    
    def __init__(self):
        self.chunker = ImprovedInsuranceClauseChunker()
        self.retriever = None 
        self.generator = EnhancedSelfReflectiveGenerator()
        self.query_analyzer = QueryAnalyzer()
        self.cache = {}
    
    def process_document(self, pdf_path: str) -> List[Document]:
        """Process document with enhanced chunking"""
        start_time = time.time()
        
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} pages from PDF")
            
            enhanced_chunks = self.chunker.smart_chunk(documents)
            print(f"Created {len(enhanced_chunks)} enhanced chunks")
            
            processing_time = time.time() - start_time
            print(f"Document processing completed in {processing_time:.2f} seconds")
            
            return enhanced_chunks
            
        except Exception as e:
            print(f"Error in document processing: {e}")
            return get_text_chunks_from_pdf(pdf_path)
    
    def create_retrieval_system(self, chunks: List[Document]):
        """Create enhanced retrieval system"""
        try:
            self.retriever = EnhancedHybridRetriever(chunks)
            print("Enhanced hybrid retriever created")
        except Exception as e:
            print(f"Error creating retrieval system: {e}")
            vector_store, ai_available = get_vector_store(chunks)
            self.retriever = vector_store
    
    @lru_cache(maxsize=50)
    def cached_query_processing(self, query_hash: str, question: str) -> Dict:
        """Cache frequent queries for better performance"""
        return self.query_analyzer.analyze_query(question)
    
    def process_question(self, question: str, max_retries: int = 3) -> EnhancedAnswer:
        """Process a single question with enhanced pipeline"""
        start_time = time.time()
        
        query_hash = hashlib.md5(question.encode()).hexdigest()
        
        if query_hash in self.cache:
            cached_result = self.cache[query_hash]
            cached_result.processing_time = time.time() - start_time
            return cached_result
        
        try:
            query_analysis = self.cached_query_processing(query_hash, question)
            
            rate_limiter.wait_if_needed()
            
            if hasattr(self.retriever, 'hybrid_search'):
                relevant_docs = self.retriever.hybrid_search(question, k=12)
            elif hasattr(self.retriever, 'similarity_search'):
                relevant_docs = self.retriever.similarity_search(question, k=5)
            else:
                relevant_docs = simple_text_search(question, self.retriever, max_results=5)
            
            if self.generator and hasattr(self.generator, 'generate_enhanced_answer'):
                enhanced_answer = self.generator.generate_enhanced_answer(
                    question, relevant_docs, query_analysis
                )
            else:
                enhanced_answer = self.generate_fallback_answer(question, relevant_docs)
            
            enhanced_answer.processing_time = time.time() - start_time
            self.cache[query_hash] = enhanced_answer
            
            return enhanced_answer
            
        except Exception as e:
            rate_limiter.handle_error(e)
            print(f"Error processing question: {e}")
            
            if max_retries > 0:
                time.sleep(1)
                return self.process_question(question, max_retries - 1)
            
            return self.generate_emergency_fallback(question)
    
    def generate_fallback_answer(self, question: str, relevant_docs: List[Document]) -> EnhancedAnswer:
        """Generate answer using fallback methods"""
        context = "\n\n".join([doc.page_content for doc in relevant_docs[:3]])
        
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

# Enhanced helper functions
def enhanced_download_pdf(url: str, save_path: str) -> bool:
    """Enhanced PDF download with better error handling and validation"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, stream=True, timeout=60, headers=headers)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
            print(f"Warning: Content type is {content_type}, may not be a PDF")
        
        content_length = response.headers.get('content-length')
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > 50:
                print(f"Warning: Large file size: {size_mb:.1f}MB")
        
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

def get_text_chunks_from_pdf(pdf_path: str) -> List[Document]:
    """Enhanced PDF chunking with better error handling"""
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        
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
        
        batch_size = 50
        if len(text_chunks) > batch_size:
            print(f"Processing {len(text_chunks)} chunks in batches of {batch_size}")
            
            vector_store = FAISS.from_documents(text_chunks[:batch_size], embedding=embeddings)
            
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
            temperature=0.05,
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

def simple_text_search(question, documents, max_results=3):
    """Simple text-based search when AI API is not available."""
    question_lower = question.lower()
    question_words = set(re.findall(r'\b\w+\b', question_lower))
    
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    question_words = question_words - stop_words
    
    scored_docs = []
    for doc in documents:
        content_lower = doc.page_content.lower()
        score = 0
        
        for word in question_words:
            if word in content_lower:
                score += 1
        
        for word in question_words:
            if word in content_lower:
                score += 0.5
        
        if score > 0:
            relevant_sentences.append((score, sentence.strip()))
    
    if relevant_sentences:
        relevant_sentences.sort(key=lambda x: x[0], reverse=True)
        return relevant_sentences[0][1]
    else:
        return "The answer is not available in the provided document."

def generate_simple_answer(question, context):
    """Generate a simple answer based on context when AI is not available."""
    if not context:
        return "The answer is not available in the provided document."
    
    question_lower = question.lower()
    
    sentences = re.split(r'[.!?]+', context)
    relevant_sentences = []
    
    question_words = set(re.findall(r'\b\w+\b', question_lower))
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
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
            "cache_size": 0  # Will be updated when rag_system is available
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
        if len(questions) > 15:
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
        
        # Create retrieval system
        rag_system.create_retrieval_system(text_chunks)
        
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
                pass

# Health monitoring endpoint
@app.route("/health/detailed", methods=["GET"])
def detailed_health_check():
    """Detailed system health check"""
    try:
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

# Test endpoint for debugging
@app.route("/test", methods=["GET"])
def test_endpoint():
    """Test endpoint to verify the app is working."""
    return jsonify({
        "message": "Enhanced HackRx API v2.0 is working!",
        "timestamp": time.time(),
        "version": "2.0-enhanced",
        "bearer_token_check": "configured"
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