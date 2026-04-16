"""
Orchestrator: Main pipeline coordinating the three agents.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.gatekeeper import Gatekeeper
from agents.verifier import Verifier
from agents.editor import Editor
import yaml


class RAGRetriever:
    """RAG retriever using sentence-transformers embeddings."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.top_k = self.config['pipeline']['rag_top_k']
        self.documents = []  # List of dicts with text, source, title
        self.doc_embeddings = None
        self.model = None
        
        # Load embedding model
        self._load_model()
        
        # Load documents
        self._load_documents()
        
        # Build embeddings index
        if self.documents:
            self._build_index()
    
    def _load_model(self):
        """Load sentence-transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Loaded embedding model: all-MiniLM-L6-v2")
        except ImportError:
            print("sentence-transformers not installed. Please run: pip install sentence-transformers")
            raise
    
    def _load_documents(self):
        """Load documents from output.json."""
        original_path = self.config['data']['original']
        
        if os.path.exists(original_path):
            with open(original_path, 'r') as f:
                data = json.load(f)
                
                # Handle the nested structure of output.json
                if "raw data" in data:
                    for source_key, source_data in data["raw data"].items():
                        if isinstance(source_data, list):
                            for item in source_data:
                                text = item.get('text', '')
                                if text:
                                    self.documents.append({
                                        "source": item.get('source', source_key),
                                        "category": item.get('category', 'unknown'),
                                        "title": item.get('title', 'untitled'),
                                        "text": text
                                    })
                # Also handle if data is directly a list
                elif isinstance(data, list):
                    for item in data:
                        text = item.get('text', '')
                        if text:
                            self.documents.append({
                                "source": item.get('source', 'unknown'),
                                "category": item.get('category', 'unknown'),
                                "title": item.get('title', 'untitled'),
                                "text": text
                            })
            
            print(f"Loaded {len(self.documents)} documents from {original_path}")
        else:
            print(f"Warning: {original_path} not found")
    
    def _build_index(self):
        """Build embeddings index for all documents."""
        if not self.documents:
            return
        
        # Extract texts and create embeddings
        texts = [doc['text'] for doc in self.documents]
        print(f"Building embeddings for {len(texts)} documents...")
        self.doc_embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Embeddings shape: {self.doc_embeddings.shape}")
    
    def retrieve(self, query: str) -> List[str]:
        """Retrieve top-k relevant passages using semantic search."""
        if not self.documents or self.doc_embeddings is None:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        similarities = cosine_similarity([query_embedding], self.doc_embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        
        # Return top-k texts
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Only include if some similarity
                results.append(self.documents[idx]['text'])
        
        return results


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
        """Retrieve relevant chunks from RAG knowledge base."""
        return self.retriever.retrieve(query)
    
    def _generate_answer(self, query: str, chunks: List[str]) -> str:
        """Generate answer using the verifier's model."""
        context = "\n\n".join(chunks) if chunks else "No context available."
        
        prompt = f"""Based on the following context, answer the user's query accurately and concisely.

Context:
{context}

Query: {query}

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
        """Process a query through the multi-agent pipeline."""
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
        
        # Step 1: Retrieve relevant chunks
        chunks = self._retrieve_chunks(query)
        result["retrieved_chunks"] = chunks
        
        # Step 2: Gatekeeper evaluates confidence
        should_stop, gatekeeper_result = self.gatekeeper.should_stop(query, chunks)
        result["gatekeeper"] = gatekeeper_result
        
        if should_stop:
            # Return "I don't know" response
            result["response"] = self.gatekeeper.get_idk_response(gatekeeper_result)
            result["final"] = True
            result["agent_stopped_at"] = "gatekeeper"
            return result
        
        # Step 3: Generate candidate answer
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
                "gatekeeper_confidence": gatekeeper_result.get("confidence"),
                "verifier_faithfulness": verifier_result.get("faithfulness"),
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


if __name__ == "__main__":
    # Test the pipeline
    pipeline = PipelineOrchestrator()
    
    # Test homogeneous configuration (Gemma only)
    print("Testing homogeneous configuration (Gemma only)...")
    pipeline.assign_models("gemma", "gemma", "gemma")
    
    result = pipeline.process_query("What is natural selection?")
    print(f"Response: {result.get('response')[:200]}...")
    print(f"Final: {result.get('final')}")
    print(f"Stopped at: {result.get('agent_stopped_at')}")
    
    # Test heterogeneous configuration
    print("\nTesting heterogeneous configuration (Gemma + Qwen + Llama)...")
    pipeline.assign_models("gemma", "qwen", "llama")
    
    result = pipeline.process_query("Explain how photosynthesis works.")
    print(f"Response: {result.get('response')[:200]}...")
