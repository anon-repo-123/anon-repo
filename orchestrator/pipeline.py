"""
Orchestrator: Main pipeline coordinating the three agents.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.gatekeeper import Gatekeeper
from agents.verifier import Verifier
from agents.editor import Editor
import yaml


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
        
        # Agent model assignments (can be homogeneous or heterogeneous)
        self.gatekeeper_model = None
        self.verifier_model = None
        self.editor_model = None
        
        # Human-in-the-loop review queue
        self.hitl_queue = []
        self.reviewed = []
        
        # RAG retrieval (simplified - in practice would use vector DB)
        self.rag_chunks = {}
        
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
        Simplified - in practice would use embeddings and vector search.
        """
        # For demo, return empty list - in production, implement actual retrieval
        # This would query a vector database with the query
        return []
    
    def _generate_answer(self, query: str, chunks: List[str]) -> str:
        """
        Generate answer using the verifier's model.
        """
        prompt = f"""Based on the following context, answer the user's query accurately and concisely.

Context:
{chr(10).join(chunks) if chunks else 'No context available.'}

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
