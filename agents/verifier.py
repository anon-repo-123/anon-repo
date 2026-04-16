"""
Verifier Agent: Fact-checks candidate responses against RAG passages before generation.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI
import yaml


class Verifier:
    """
    Verifier agent that fact-checks candidate responses against retrieved passages.
    Computes faithfulness scores and determines if response is supported.
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
        
        self.faithfulness_threshold = self.config['verifier']['faithfulness_threshold']
        self.model_name = None
        
    def set_model(self, model_key: str):
        """Assign a model to this agent from config."""
        model_config = self.config['models'].get(model_key.lower())
        if not model_config:
            raise ValueError(f"Model {model_key} not found in config")
        self.model_name = model_config['name']
        self.supports_reasoning = model_config.get('supports_reasoning', False)
        self.model_key = model_key
    
    def _get_faithfulness_prompt(self, answer: str, passages: List[str]) -> str:
        """Generate prompt for faithfulness evaluation."""
        context = "\n\n---\n\n".join(passages)
        
        return f"""You are a Verifier agent for an educational system. Your task is to check if the generated answer is faithful to (supported by) the retrieved passages.

Retrieved Passages:
{context}

Generated Answer:
{answer}

Rate the faithfulness of the answer (0.0 to 1.0) based on whether each claim in the answer can be directly supported by the passages.

Guidelines:
- 1.0: Every claim in the answer is directly stated or clearly implied by the passages
- 0.8-0.9: Most claims are supported, minor unsupported details
- 0.6-0.7: Some claims are supported, others are not
- 0.4-0.5: Few claims are supported, major hallucinations present
- 0.0-0.3: Answer is largely or completely unsupported

Consider:
- Does the answer introduce facts not in the passages?
- Are there contradictions between answer and passages?
- Is the answer missing critical context from passages?

Respond with ONLY a valid JSON object in this exact format:
{{"faithfulness": 0.85, "reason": "brief explanation", "unsupported_claims": ["claim1", "claim2"]}}

Do not include any text outside the JSON object."""
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from model response."""
        if content is None:
            return {"faithfulness": 0.5, "reason": "No response content", "unsupported_claims": []}
        
        json_pattern = r'\{[^{}]*"faithfulness"\s*:\s*[0-9.]+[^{}]*\}'
        match = re.search(json_pattern, content)
        
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                potential_json = content[start:end]
                return json.loads(potential_json)
        except json.JSONDecodeError:
            pass
        
        faithfulness_match = re.search(r'faithfulness["\s:]+([0-9.]+)', content, re.IGNORECASE)
        if faithfulness_match:
            faithfulness = float(faithfulness_match.group(1))
            return {"faithfulness": faithfulness, "reason": "Extracted from response", "unsupported_claims": []}
        
        return {"faithfulness": 0.5, "reason": "Could not parse response", "unsupported_claims": []}
    
    def evaluate_faithfulness(self, answer: str, passages: List[str]) -> Dict[str, Any]:
        """
        Evaluate faithfulness of answer against retrieved passages.
        
        Returns:
            Dict with keys: faithfulness, reason, unsupported_claims
        """
        prompt = self._get_faithfulness_prompt(answer, passages)
        
        messages = [{"role": "user", "content": prompt}]
        
        api_args = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.config['pipeline']['temperature'],
            "max_tokens": 500,
        }
        
        if self.supports_reasoning:
            api_args["extra_body"] = {"reasoning": {"enabled": True}}
        
        try:
            response = self.client.chat.completions.create(**api_args)
            
            message = response.choices[0].message
            content = message.content
            
            if content is None and hasattr(message, 'reasoning'):
                content = message.reasoning
            
            if content is None and hasattr(message, 'reasoning_details') and message.reasoning_details:
                if isinstance(message.reasoning_details, list) and len(message.reasoning_details) > 0:
                    content = message.reasoning_details[0].get('text', '')
            
            result = self._parse_json_response(content if content else "")
            
            faithfulness = float(result.get("faithfulness", 0.5))
            faithfulness = max(0.0, min(1.0, faithfulness))
            
            return {
                "faithfulness": faithfulness,
                "reason": result.get("reason", "No reason provided"),
                "unsupported_claims": result.get("unsupported_claims", [])
            }
            
        except Exception as e:
            print(f"Verifier evaluation error: {e}")
            return {
                "faithfulness": 0.5,
                "reason": f"Evaluation failed: {str(e)}",
                "unsupported_claims": ["Unable to evaluate faithfulness"]
            }
    
    def is_faithful(self, answer: str, passages: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if answer is faithful enough based on threshold.
        
        Returns:
            (is_faithful, evaluation_result)
        """
        eval_result = self.evaluate_faithfulness(answer, passages)
        faithfulness = eval_result["faithfulness"]
        
        is_faithful = faithfulness >= self.faithfulness_threshold
        
        return is_faithful, eval_result


if __name__ == "__main__":
    verifier = Verifier()
    verifier.set_model("llama")
    
    test_answer = "Natural selection is the differential survival and reproduction of individuals due to differences in phenotype."
    test_passages = [
        "Natural selection is the differential survival and reproduction of individuals due to differences in phenotype.",
        "Charles Darwin popularized the term 'natural selection' in his 1859 book On the Origin of Species."
    ]
    
    is_faithful, eval_result = verifier.is_faithful(test_answer, test_passages)
    print(f"Is faithful: {is_faithful}")
    print(f"Evaluation: {json.dumps(eval_result, indent=2)}")
