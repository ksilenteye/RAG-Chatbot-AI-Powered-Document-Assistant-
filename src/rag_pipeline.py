import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import time
from typing import List, Tuple, Dict
import numpy as np

class ImprovedRAGPipeline:
    
    def __init__(self,
                 vector_db_path="vectordb/index.faiss",
                 chunk_path="vectordb/chunks.pkl",
                 embedding_model="all-MiniLM-L6-v2",
                 llm_model="microsoft/DialoGPT-medium"):
        try:
            print("Loading FAISS index and chunks...")
            if not os.path.exists(vector_db_path):
                raise FileNotFoundError(f"FAISS index not found at {vector_db_path}")
            self.index = faiss.read_index(vector_db_path)

            if not os.path.exists(chunk_path):
                raise FileNotFoundError(f"Chunks file not found at {chunk_path}")
            with open(chunk_path, "rb") as f:
                self.chunks = pickle.load(f)

            print("Loading embedding model...")
            self.embedder = SentenceTransformer(embedding_model)

            print("Loading LLM (may take time)...")
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
            self.model = AutoModelForCausalLM.from_pretrained(llm_model)
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            self.model.eval()
            if self.device == "cuda":
                self.model.half()
            
            self.embedding_cache = {}
            
            print(f"RAG Pipeline initialized successfully on {self.device}!")
            
        except Exception as e:
            print(f"Error initializing RAG pipeline: {e}")
            raise

    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        try:
            if query in self.embedding_cache:
                query_embedding = self.embedding_cache[query]
            else:
                query_embedding = self.embedder.encode([query])
                self.embedding_cache[query] = query_embedding
            
            D, I = self.index.search(query_embedding, k)
            
            similarity_scores = 1 / (1 + D[0])
            
            retrieved_chunks_with_scores = []
            for i, score in zip(I[0], similarity_scores):
                if i < len(self.chunks):
                    retrieved_chunks_with_scores.append((self.chunks[i], score))
            
            print(f"Retrieved {len(retrieved_chunks_with_scores)} chunks with scores")
            return retrieved_chunks_with_scores
            
        except Exception as e:
            print(f"Error in retrieval: {e}")
            return []

    def filter_relevant_chunks(self, chunks_with_scores: List[Tuple[str, float]], 
                             threshold: float = 0.3) -> List[str]:
        relevant_chunks = []
        for chunk, score in chunks_with_scores:
            if score >= threshold:
                relevant_chunks.append(chunk)
        
        print(f"Filtered to {len(relevant_chunks)} relevant chunks (threshold: {threshold})")
        return relevant_chunks

    def build_improved_prompt(self, question: str, retrieved_chunks: List[str]) -> str:
        try:
            context = "\n\n".join(retrieved_chunks)
            
            prompt = f"""You are a helpful AI assistant. Answer the user's question based ONLY on the provided context. If the context doesn't contain enough information to answer the question accurately, say "I don't have enough information to answer this question accurately."

IMPORTANT RULES:
1. Only use information from the provided context
2. If you're not sure about something, say so
3. Be concise but comprehensive
4. Cite specific parts of the context when possible

Context:
{context}

Question: {question}

Answer:"""
            return prompt
        except Exception as e:
            print(f"Error building prompt: {e}")
            return f"Question: {question}\nAnswer: I apologize, but I encountered an error processing your question."

    def generate_with_confidence(self, prompt: str, max_tokens: int = 512) -> Tuple[str, float]:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens, 
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            answer = generated_text.replace(prompt, "").strip()
            
            confidence = min(1.0, len(answer) / 100)
            
            return answer, confidence
            
        except Exception as e:
            print(f"Error in generation: {e}")
            return "Sorry, I encountered an error while generating the answer.", 0.0

    def detect_hallucinations(self, answer: str, context: str) -> Dict[str, any]:
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        overlap = len(answer_words.intersection(context_words))
        total_answer_words = len(answer_words)
        
        overlap_ratio = overlap / total_answer_words if total_answer_words > 0 else 0
        
        is_potential_hallucination = overlap_ratio < 0.2
        
        return {
            "overlap_ratio": overlap_ratio,
            "is_potential_hallucination": is_potential_hallucination,
            "confidence": overlap_ratio
        }

    def query(self, question: str, k: int = 5, max_tokens: int = 512, 
              similarity_threshold: float = 0.3) -> Dict[str, any]:
        start_time = time.time()
        
        try:
            print(f"Processing query: {question}")
            
            chunks_with_scores = self.retrieve_with_scores(question, k)
            
            if not chunks_with_scores:
                return {
                    "answer": "No relevant information found.",
                    "chunks": [],
                    "confidence": 0.0,
                    "processing_time": time.time() - start_time,
                    "hallucination_detection": {"is_potential_hallucination": False, "confidence": 0.0}
                }
            
            relevant_chunks = self.filter_relevant_chunks(chunks_with_scores, similarity_threshold)
            
            if not relevant_chunks:
                return {
                    "answer": "The available information doesn't seem relevant to your question.",
                    "chunks": [],
                    "confidence": 0.0,
                    "processing_time": time.time() - start_time,
                    "hallucination_detection": {"is_potential_hallucination": False, "confidence": 0.0}
                }
            
            prompt = self.build_improved_prompt(question, relevant_chunks)
            answer, confidence = self.generate_with_confidence(prompt, max_tokens)
            
            context_text = "\n".join(relevant_chunks)
            hallucination_results = self.detect_hallucinations(answer, context_text)
            
            processing_time = time.time() - start_time
            
            return {
                "answer": answer,
                "chunks": relevant_chunks,
                "chunks_with_scores": chunks_with_scores,
                "confidence": confidence,
                "processing_time": processing_time,
                "hallucination_detection": hallucination_results
            }
            
        except Exception as e:
            print(f"Error in query processing: {e}")
            return {
                "answer": "Sorry, I encountered an error while processing your query.",
                "chunks": [],
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "hallucination_detection": {"is_potential_hallucination": False, "confidence": 0.0}
            }

    def get_performance_stats(self) -> Dict[str, any]:
        return {
            "device": self.device,
            "total_chunks": len(self.chunks),
            "embedding_cache_size": len(self.embedding_cache),
            "model_name": self.model.config._name_or_path
        }

def test_pipeline():
    try:
        rag = ImprovedRAGPipeline()
        
        test_questions = [
            "What is machine learning?",
            "How do neural networks work?",
            "What are the applications of AI in healthcare?"
        ]
        
        for question in test_questions:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print(f"{'='*60}")
            
            result = rag.query(question)
            
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Processing Time: {result['processing_time']:.2f}s")
            print(f"Potential Hallucination: {result['hallucination_detection']['is_potential_hallucination']}")
            print(f"Hallucination Confidence: {result['hallucination_detection']['confidence']:.2f}")
            
            print(f"\nRetrieved {len(result['chunks'])} chunks")
            for i, chunk in enumerate(result['chunks'][:2]):
                print(f"Chunk {i+1}: {chunk[:100]}...")
        
        stats = rag.get_performance_stats()
        print(f"\n{'='*60}")
        print("PERFORMANCE STATS")
        print(f"{'='*60}")
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error testing pipeline: {e}")

if __name__ == "__main__":
    test_pipeline() 