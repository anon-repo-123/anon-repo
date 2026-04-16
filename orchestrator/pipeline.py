"""
Orchestrator: Main pipeline coordinating the three agents with RAG retrieval.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.gatekeeper import Gatekeeper
from agents.verifier import Verifier
from agents.editor import Editor
import yaml


class RAGRetriever:
    """RAG retriever for fetching relevant passages from knowledge base."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.chunk_size = self.config['pipeline']['rag_chunk_size']
        self.top_k = self.config['pipeline']['rag_top_k']
        self.documents = []  # List of (source, chunk_text, metadata)
        self.chunk_embeddings = None
        self.tfidf_vectorizer = None
        
        # Load knowledge base
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load and chunk the knowledge base from data files."""
        documents = []
        
        # Load original data
        original_path = self.config['data']['original']
        if os.path.exists(original_path):
            with open(original_path, 'r') as f:
                data = json.load(f)
                if "raw data" in data:
                    for source_key, source_data in data["raw data"].items():
                        if isinstance(source_data, list):
                            for item in source_data:
                                text = item.get('text', '')
                                if text:
                                    chunks = self._chunk_text(text)
                                    for chunk in chunks:
                                        documents.append({
                                            "source": item.get('source', source_key),
                                            "category": item.get('category', 'unknown'),
                                            "title": item.get('title', 'untitled'),
                                            "text": chunk,
                                            "chunk_id": len(documents)
                                        })
        
        # Load hallucinated data (original passages only)
        hallu_path = self.config['data']['hallucinated']
        if os.path.exists(hallu_path):
            with open(hallu_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    original = item.get('original', '')
                    if original:
                        chunks = self._chunk_text(original)
                        for chunk in chunks:
                            documents.append({
                                "source": item.get('source', 'unknown'),
                                "category": item.get('category', 'unknown'),
                                "title": item.get('title', 'untitled'),
                                "error_type": item.get('error_type', 'none'),
                                "text": chunk,
                                "chunk_id": len(documents)
                            })
        
        self.documents = documents
        print(f"Loaded {len(self.documents)} document chunks into knowledge base")
        
        # Build search index
        if self.documents:
            self._build_search_index()
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []
        
        # Simple sentence-based chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            if current_length + sentence_len > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep overlap: last sentence carries over
                overlap = current_chunk[-1] if current_chunk else ''
                current_chunk = [overlap] if overlap else []
                current_length = len(overlap)
            
            current_chunk.append(sentence)
            current_length += sentence_len
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text[:self.chunk_size]]
    
    def _build_search_index(self):
        """Build search index for document retrieval."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Extract all document texts
            doc_texts = [doc['text'] for doc in self.documents]
            
            # Build TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.chunk_embeddings = self.tfidf_vectorizer.fit_transform(doc_texts)
            self.cosine_similarity = cosine_similarity
            print(f"Built TF-IDF index with {self.chunk_embeddings.shape[1]} features")
            
        except ImportError:
            print("scikit-learn not available. Using fallback keyword matching.")
            self.tfidf_vectorizer = None
            self.chunk_embeddings = None
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant passages for a query."""
        if top_k is None:
            top_k = self.top_k
        
        if not self.documents:
            return []
        
        # Use TF-IDF if available
        if self.tfidf_vectorizer is not None and self.chunk_embeddings is not None:
            try:
                # Transform query to TF-IDF vector
                query_vec = self.tfidf_vectorizer.transform([query])
                
                # Compute cosine similarity
                similarities = self.cosine_similarity(query_vec, self.chunk_embeddings).flatten()
                
                # Get top-k indices
                top_indices = similarities.argsort()[-top_k:][::-1]
                
                results = []
                for idx in top_indices:
                    if similarities[idx] > 0.05:  # Only include if some similarity
                        doc = self.documents[idx]
                        results.append({
                            "text": doc['text'],
                            "source": doc.get('source', 'unknown'),
                            "title": doc.get('title', 'untitled'),
                            "score": float(similarities[idx]),
                            "chunk_id": doc.get('chunk_id', idx)
                        })
                
                return results
                
            except Exception as e:
                print(f"TF-IDF retrieval error: {e}, falling back to keyword matching")
        
        # Fallback: simple keyword matching
        return self._keyword_retrieve(query, top_k)
    
    def _keyword_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Simple keyword-based retrieval fallback."""
        query_words = set(query.lower().split())
        
        scores = []
        for doc in self.documents:
            doc_words = set(doc['text'].lower().split())
            overlap = len(query_words & doc_words)
            score = overlap / (len(query_words) + len(doc_words) - overlap + 1)
            scores.append(score)
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.documents[idx]
                results.append({
                    "text": doc['text'],
                    "source": doc.get('source', 'unknown'),
                    "title": doc.get('title', 'untitled'),
                    "score": scores[idx],
                    "chunk_id": doc.get('chunk_id', idx)
                })
        
        return results
    
    def get_chunk_texts(self, query: str, top_k: int = None) -> List[str]:
        """Get just the text of retrieved chunks."""
        results = self.retrieve(query, top_k)
        return [r['text'] for r in results]


class PipelineOrchestrator:
    """
    Orchestrates the multi-agent pipeline: Gatekeeper -> Verifier -> Editor
    with human-in-the-loop fallback for uncertain cases.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize agents
        self.gatekeeper = Gatekeeper(config_path)
        self.verifier = Verifier(config_path)
        self.editor = Editor(config_path)
        
        # Initialize RAG retriever
        self.retriever = RAGRetriever(config_path)
        
        # Agent model assignments (can be homogeneous or heterogeneous)
        self.gatekeeper_model = None
        self.verifier_model = None
        self.editor_model = None
        
        # Human-in-the-loop review queue
        self.hitl_queue = []
        self.reviewed = []
        
    def assign_models(self, gatekeeper_model: str, verifier_model: str, editor_model: str):
        """Assign specific models to each agent."""
        self.gatekeeper_model = gatekeeper_model.lower()
        self.verifier_model = verifier_model.lower()
        self.editor_model = editor_model.lower()
        
        self.gatekeeper.set_model(gatekeeper_model)
        self.verifier.set_model(verifier_model)
        self.editor.set_model(editor_model)
        
    def _retrieve_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve relevant chunks from RAG knowledge base.
        Now fully implemented with TF-IDF retrieval.
        """
        return self.retriever.get_chunk_texts(query, top_k)
    
    def _retrieve_chunks_with_metadata(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks with metadata for debugging."""
        return self.retriever.retrieve(query, top_k)
    
    def _generate_answer(self, query: str, chunks: List[str]) -> str:
        """
        Generate answer using the verifier's model with RAG context.
        """
        if not chunks:
            context = "No specific context retrieved. Answer based on general knowledge, but note that this response may not be verified against educational materials."
        else:
            context = "\n\n---\n\n".join(chunks)
        
        prompt = f"""You are an educational assistant. Answer the following query based primarily on the provided context. If the context contains the answer, use it directly. If the context is insufficient, state that you are uncertain.

Context:
{context}

Query: {query}

Instructions:
1. Be accurate and concise
2. Only include information supported by the context when possible
3. If you must use external knowledge, indicate uncertainty

Answer:"""
        
        # Use verifier's model to generate answer
        api_args = {
            "model": self.verifier.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config['pipeline']['temperature'],
            "max_tokens": self.config['pipeline']['max_tokens'],
        }
        
        if self.verifier.supports_reasoning:
            api_args["extra_body"] = {"reasoning": {"enabled": True}}
        
        try:
            response = self.verifier.client.chat.completions.create(**api_args)
            content = response.choices[0].message.content
            
            if content is None and hasattr(response.choices[0].message, 'reasoning'):
                content = response.choices[0].message.reasoning
                
            return content if content else "Unable to generate response."
        except Exception as e:
            print(f"Answer generation error: {e}")
            return f"Error generating answer: {str(e)}"
    
    def _needs_hitl_review(self, gatekeeper_result: Dict, verifier_result: Dict = None, 
                          editor_metadata: Dict = None) -> Tuple[bool, str]:
        """Determine if human review is needed."""
        confidence = gatekeeper_result.get("confidence", 0.5)
        
        # Case 1: Confidence below 0.5
        if confidence < 0.5:
            return True, "low_confidence_below_0.5"
        
        # Case 2: Confidence between 0.5 and 0.75 with repeated query
        if 0.5 <= confidence < 0.75:
            return True, "medium_confidence_requires_review"
        
        # Case 3: Faithfulness below threshold
        if verifier_result and verifier_result.get("faithfulness", 1.0) < 0.7:
            return True, "low_faithfulness"
        
        # Case 4: Editor removed more than 50% of content
        if editor_metadata and editor_metadata.get("removal_percentage", 0) > 0.5:
            return True, "excessive_removal"
        
        return False, None
    
    def process_query(self, query: str, query_id: str = None) -> Dict[str, Any]:
        """
        Process a query through the multi-agent pipeline.
        
        Returns:
            Dictionary with response and all intermediate results
        """
        if query_id is None:
            query_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        result = {
            "query_id": query_id,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "gatekeeper_model": self.gatekeeper_model,
                "verifier_model": self.verifier_model,
                "editor_model": self.editor_model
            }
        }
        
        # Step 1: Retrieve relevant chunks with metadata
        chunks = self._retrieve_chunks(query)
        chunks_metadata = self._retrieve_chunks_with_metadata(query)
        
        result["retrieved_chunks"] = chunks_metadata
        result["retrieved_chunk_count"] = len(chunks)
        
        # Step 2: Gatekeeper evaluates confidence
        should_stop, gatekeeper_result = self.gatekeeper.should_stop(query, chunks)
        result["gatekeeper"] = gatekeeper_result
        
        if should_stop:
            # Return "I don't know" response
            result["response"] = self.gatekeeper.get_idk_response(gatekeeper_result)
            result["final"] = True
            result["agent_stopped_at"] = "gatekeeper"
            return result
        
        # Step 3: Generate candidate answer using RAG
        candidate_answer = self._generate_answer(query, chunks)
        result["candidate_answer"] = candidate_answer
        
        # Step 4: Verifier fact-checks
        is_faithful, verifier_result = self.verifier.is_faithful(candidate_answer, chunks)
        result["verifier"] = verifier_result
        
        # Step 5: Editor compresses response
        confidence = gatekeeper_result.get("confidence", 1.0)
        edited_answer, editor_metadata = self.editor.edit(candidate_answer, confidence)
        result["editor"] = editor_metadata
        result["edited_answer"] = edited_answer
        
        # Step 6: Check if human review needed
        needs_review, review_reason = self._needs_hitl_review(
            gatekeeper_result, verifier_result, editor_metadata
        )
        result["needs_hitl_review"] = needs_review
        result["review_reason"] = review_reason
        
        if needs_review:
            # Add to HITL queue
            hitl_entry = {
                "query_id": query_id,
                "query": query,
                "candidate_answer": candidate_answer,
                "edited_answer": edited_answer,
                "retrieved_chunks": chunks_metadata,
                "gatekeeper_confidence": gatekeeper_result.get("confidence"),
                "gatekeeper_reason": gatekeeper_result.get("reason"),
                "verifier_faithfulness": verifier_result.get("faithfulness"),
                "verifier_unsupported_claims": verifier_result.get("unsupported_claims", []),
                "removal_percentage": editor_metadata.get("removal_percentage"),
                "review_reason": review_reason,
                "status": "pending"
            }
            self.hitl_queue.append(hitl_entry)
            result["response"] = "Response requires human review. Please check the dashboard."
            result["final"] = False
        else:
            result["response"] = edited_answer
            result["final"] = True
        
        result["agent_stopped_at"] = "editor" if not needs_review else "hitl"
        
        return result
    
    def save_hitl_queue(self, filepath: str = "results/hitl_queue.json"):
        """Save HITL queue to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.hitl_queue, f, indent=2)
    
    def load_hitl_queue(self, filepath: str = "results/hitl_queue.json"):
        """Load HITL queue from file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.hitl_queue = json.load(f)
    
    def get_pending_reviews(self) -> List[Dict]:
        """Get all pending HITL reviews."""
        return [item for item in self.hitl_queue if item.get("status") == "pending"]
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        return {
            "total_chunks": len(self.retriever.documents),
            "chunk_size": self.retriever.chunk_size,
            "retrieval_top_k": self.retriever.top_k,
            "has_tfidf": self.retriever.tfidf_vectorizer is not None
        }


if __name__ == "__main__":
    # Test the pipeline with RAG
    pipeline = PipelineOrchestrator()
    
    # Show knowledge base stats
    stats = pipeline.get_knowledge_base_stats()
    print(f"\nKnowledge Base Stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Chunk size: {stats['chunk_size']}")
    print(f"  TF-IDF available: {stats['has_tfidf']}")
    
    # Test homogeneous configuration (Gemma only)
    print()
    print("Testing homogeneous configuration (Gemma only)...")
    print()
    pipeline.assign_models("gemma", "gemma", "gemma")
    
    result = pipeline.process_query("What is natural selection?")
    print(f"\nRetrieved {result['retrieved_chunk_count']} chunks")
    print(f"Gatekeeper confidence: {result['gatekeeper'].get('confidence', 0):.3f}")
    print(f"Response: {result.get('response')[:200]}...")
    print(f"Final: {result.get('final')}")
    print(f"Stopped at: {result.get('agent_stopped_at')}")
    
    # Test heterogeneous configuration
    print()
    print("Testing heterogeneous configuration (Gemma + Qwen + Llama)...")
    print()
    pipeline.assign_models("gemma", "qwen", "llama")
    
    result = pipeline.process_query("Explain how photosynthesis works.")
    print(f"\nRetrieved {result['retrieved_chunk_count']} chunks")
    print(f"Gatekeeper confidence: {result['gatekeeper'].get('confidence', 0):.3f}")
    if result.get('verifier'):
        print(f"Verifier faithfulness: {result['verifier'].get('faithfulness', 0):.3f}")
    print(f"Response: {result.get('response')[:200]}...")
