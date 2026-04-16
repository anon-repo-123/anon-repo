"""
Gatekeeper Agent: Stops when retrieval confidence is low and discloses uncertainty.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI
import yaml


class Gatekeeper:
    """
    Gatekeeper agent that evaluates confidence in retrieved context and query.
    Stops generation when confidence is below threshold and returns "I don't know".
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        self.client = OpenAI(
            base_url=self.config['openrouter']['base_url'],
            api_key=self.api_key,
        )
        
        self.confidence_threshold = self.config['gatekeeper']['confidence_threshold']
        self.low_confidence_threshold = self.config['gatekeeper']['low_confidence_threshold']
        self.model_name = None  # Will be set when agent is assigned a model
        
    def set_model(self, model_key: str):
        """Assign a model to this agent from config."""
        model_config = self.config['models'].get(model_key.lower())
        if not model_config:
            raise ValueError(f"Model {model_key} not found in config")
        self.model_name = model_config['name']
        self.supports_reasoning = model_config.get('supports_reasoning', False)
        self.model_key = model_key
        
    def _get_confidence_prompt(self, query: str, retrieved_chunks: List[str]) -> str:
        """Generate prompt for confidence evaluation."""
        context = "\n\n".join(retrieved_chunks[:3]) if retrieved_chunks else "No retrieved context available."
        
        return f"""You are a Gatekeeper agent for an educational system. Your task is to evaluate whether you have sufficient, reliable information to answer the user's query based on the retrieved context.

Query: {query}

Retrieved Context:
{context}

Rate your confidence (0.0 to 1.0) in being able to provide a correct, complete, and reliable answer to this query using ONLY the information in the retrieved context.

Consider:
- Does the context directly address the query?
- Is the information complete and unambiguous?
- Are there contradictions or gaps in the context?
- Would you need to infer or guess?

Respond with ONLY a valid JSON object in this exact format:
{{"confidence": 0.85, "reason": "brief explanation", "knowledge_gaps": ["gap1", "gap2"]}}

Do not include any text outside the JSON object."""
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from model response, handling various formats."""
        if content is None:
            return {"confidence": 0.5, "reason": "No response content", "knowledge_gaps": []}
        
        # Try to extract JSON from the content
        json_pattern = r'\{[^{}]*"confidence"\s*:\s*[0-9.]+[^{}]*\}'
        match = re.search(json_pattern, content)
        
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        # Try more aggressive extraction
        try:
            # Find anything that looks like JSON
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                potential_json = content[start:end]
                return json.loads(potential_json)
        except json.JSONDecodeError:
            pass
        
        # Fallback: look for confidence value
        confidence_match = re.search(r'confidence["\s:]+([0-9.]+)', content, re.IGNORECASE)
        if confidence_match:
            confidence = float(confidence_match.group(1))
            return {"confidence": confidence, "reason": "Extracted from response", "knowledge_gaps": []}
        
        # Default fallback
        return {"confidence": 0.5, "reason": "Could not parse response", "knowledge_gaps": []}
    
    def evaluate_confidence(self, query: str, retrieved_chunks: List[str]) -> Dict[str, Any]:
        """
        Evaluate confidence in answering query based on retrieved context.
        
        Returns:
            Dict with keys: confidence, reason, knowledge_gaps
        """
        prompt = self._get_confidence_prompt(query, retrieved_chunks)
        
        messages = [{"role": "user", "content": prompt}]
        
        # Prepare API call arguments
        api_args = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.config['pipeline']['temperature'],
            "max_tokens": 500,
        }
        
        # Add reasoning if supported
        if self.supports_reasoning:
            api_args["extra_body"] = {"reasoning": {"enabled": True}}
        
        try:
            response = self.client.chat.completions.create(**api_args)
            
            # Extract content - handle both regular and reasoning responses
            message = response.choices[0].message
            content = message.content
            
            # For models with reasoning, content might be in a different field
            if content is None and hasattr(message, 'reasoning'):
                content = message.reasoning
            
            if content is None:
                # Try to get from reasoning_details
                if hasattr(message, 'reasoning_details') and message.reasoning_details:
                    if isinstance(message.reasoning_details, list) and len(message.reasoning_details) > 0:
                        content = message.reasoning_details[0].get('text', '')
            
            result = self._parse_json_response(content if content else "")
            
            # Ensure confidence is within bounds
            confidence = float(result.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            
            return {
                "confidence": confidence,
                "reason": result.get("reason", "No reason provided"),
                "knowledge_gaps": result.get("knowledge_gaps", [])
            }
            
        except Exception as e:
            print(f"Gatekeeper evaluation error: {e}")
            return {
                "confidence": 0.5,
                "reason": f"Evaluation failed: {str(e)}",
                "knowledge_gaps": ["Unable to evaluate confidence"]
            }
    
    def should_stop(self, query: str, retrieved_chunks: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if system should stop and return "I don't know".
        
        Returns:
            (should_stop, evaluation_result)
        """
        eval_result = self.evaluate_confidence(query, retrieved_chunks)
        confidence = eval_result["confidence"]
        
        should_stop = confidence < self.confidence_threshold
        
        return should_stop, eval_result
    
    def get_idk_response(self, eval_result: Dict[str, Any]) -> str:
        """Generate appropriate 'I don't know' response based on confidence level."""
        confidence = eval_result["confidence"]
        gaps = eval_result.get("knowledge_gaps", [])
        
        if confidence < self.low_confidence_threshold:
            gaps_text = ", ".join(gaps[:2]) if gaps else "the requested information"
            return f"I don't have sufficient reliable information to answer this question. The retrieved knowledge base lacks {gaps_text}. Please consult your instructor or verified educational materials."
        else:
            return f"I'm uncertain about this answer (confidence: {confidence:.2f}). The available information may be incomplete or ambiguous. I recommend verifying with authoritative sources."


if __name__ == "__main__":
    # Test the gatekeeper agent
    gatekeeper = Gatekeeper()
    gatekeeper.set_model("gemma")  # Test with Gemma
    
    test_query = "What is natural selection?"
    test_chunks = [
        "Natural selection is the differential survival and reproduction of individuals due to differences in phenotype.",
        "Charles Darwin popularized the term 'natural selection' in his 1859 book On the Origin of Species.",
        "Natural selection acts on the heritable traits of organisms."
    ]
    
    should_stop, eval_result = gatekeeper.should_stop(test_query, test_chunks)
    print(f"Should stop: {should_stop}")
    print(f"Evaluation: {json.dumps(eval_result, indent=2)}")
    
    if should_stop:
        print(f"Response: {gatekeeper.get_idk_response(eval_result)}")
