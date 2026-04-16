"""
Editor Agent: Removes unimportant, verbose, or distracting remarks.
"""

import os
import json
import re
from typing import Dict, Any, List, Tuple
from openai import OpenAI
import yaml


class Editor:
    """
    Editor agent that compresses responses by removing verbose or unimportant remarks.
    Enforces sentence limits for low-confidence responses.
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
        
        self.max_sentences = self.config['editor']['max_sentences_low_confidence']
        self.confidence_threshold = self.config['editor']['confidence_threshold_for_compression']
        self.max_removal_percentage = self.config['editor']['max_removal_percentage']
        self.model_name = None
        
    def set_model(self, model_key: str):
        """Assign a model to this agent from config."""
        model_config = self.config['models'].get(model_key.lower())
        if not model_config:
            raise ValueError(f"Model {model_key} not found in config")
        self.model_name = model_config['name']
        self.supports_reasoning = model_config.get('supports_reasoning', False)
        self.model_key = model_key
    
    def _count_sentences(self, text: str) -> int:
        """Count number of sentences in text."""
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])
    
    def _get_edit_prompt(self, answer: str, is_low_confidence: bool = False) -> str:
        """Generate prompt for editing/compression."""
        if is_low_confidence:
            return f"""You are an Editor agent for an educational system. Compress the following answer to at most {self.max_sentences} sentences while preserving ALL factual information.

Original Answer:
{answer}

Guidelines:
- Keep only the most important factual statements
- Remove hedging language ("I think", "perhaps", "maybe")
- Remove redundant explanations
- Remove qualifying remarks and caveats when they distract from core facts
- Keep all factual claims that are essential to answering the question

Respond with ONLY the compressed answer, no explanations."""
        else:
            return f"""You are an Editor agent for an educational system. Remove unimportant, verbose, or distracting remarks from the following answer.

Original Answer:
{answer}

Guidelines:
- Remove redundant phrases
- Remove unnecessary hedging ("very", "quite", "somewhat")
- Remove off-topic remarks
- Keep all factual information and key explanations
- Make the response more concise without losing meaning

Respond with ONLY the edited answer, no explanations."""
    
    def _edit_direct(self, answer: str, is_low_confidence: bool = False) -> str:
        """Simple rule-based editing fallback."""
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        
        if is_low_confidence:
            return ' '.join(sentences[:self.max_sentences])
        
        # Remove sentences with hedging
        hedging_patterns = [r'\bI think\b', r'\bperhaps\b', r'\bmaybe\b', r'\bquite\b', r'\bvery\b']
        edited = answer
        for pattern in hedging_patterns:
            edited = re.sub(pattern, '', edited, flags=re.IGNORECASE)
        
        # Remove redundant whitespace
        edited = re.sub(r'\s+', ' ', edited).strip()
        
        return edited if edited else answer
    
    def edit(self, answer: str, gatekeeper_confidence: float = 1.0) -> Tuple[str, Dict[str, Any]]:
        """
        Edit/compress the answer based on confidence level.
        
        Returns:
            (edited_answer, metadata) where metadata includes original_length, new_length, removal_percentage
        """
        original_length = len(answer)
        is_low_confidence = gatekeeper_confidence < self.confidence_threshold
        
        prompt = self._get_edit_prompt(answer, is_low_confidence)
        
        messages = [{"role": "user", "content": prompt}]
        
        api_args = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.config['pipeline']['temperature'],
            "max_tokens": 512,
        }
        
        if self.supports_reasoning:
            api_args["extra_body"] = {"reasoning": {"enabled": True}}
        
        try:
            response = self.client.chat.completions.create(**api_args)
            
            message = response.choices[0].message
            edited_answer = message.content
            
            if edited_answer is None and hasattr(message, 'reasoning'):
                edited_answer = message.reasoning
            
            if edited_answer is None:
                edited_answer = self._edit_direct(answer, is_low_confidence)
            
            # Ensure sentence limit for low confidence
            if is_low_confidence and self._count_sentences(edited_answer) > self.max_sentences:
                edited_answer = self._edit_direct(answer, is_low_confidence)
            
            new_length = len(edited_answer)
            removal_percentage = 1 - (new_length / original_length) if original_length > 0 else 0
            
            metadata = {
                "original_length": original_length,
                "new_length": new_length,
                "removal_percentage": removal_percentage,
                "was_compressed": is_low_confidence,
                "exceeds_removal_threshold": removal_percentage > self.max_removal_percentage
            }
            
            return edited_answer, metadata
            
        except Exception as e:
            print(f"Editor error: {e}")
            edited_answer = self._edit_direct(answer, is_low_confidence)
            return edited_answer, {
                "original_length": original_length,
                "new_length": len(edited_answer),
                "removal_percentage": 1 - (len(edited_answer) / original_length) if original_length > 0 else 0,
                "error": str(e)
            }


if __name__ == "__main__":
    editor = Editor()
    editor.set_model("mistral")
    
    test_answer = "Natural selection is, I think, the differential survival and reproduction of individuals due to differences in phenotype. It was, perhaps, first described by Charles Darwin. This is a very important concept in evolutionary biology. The mechanism acts on heritable traits over generations."
    
    edited_answer, metadata = editor.edit(test_answer, gatekeeper_confidence=0.8)
    print(f"Original: {test_answer}")
    print(f"Edited: {edited_answer}")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")
