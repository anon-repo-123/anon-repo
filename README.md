# Multi-Agent Hallucination Detection for Educational LLMs

A multi-agent framework for detecting and suppressing hallucinatory fluency in LLM responses for educational contexts. The system implies specialized agents (Gatekeeper, Verifier, Editor) in sequence with human-in-the-loop fallback when confidence is low.

### Agents

```
Query → Gatekeeper (confidence check) → Verifier (fact-check) → Editor (compress) → Response
              ↓                              ↓                      ↓
         "I don't know" if              Human review if        3-sentence max if
         confidence < 0.75              faithfulness < 0.70    confidence < 0.9
```

* Gatekeeper stops when retrieval confidence is low
* Verifier fact-checks against RAG passages 
* Editor removes verbose/unimportant remarks

### Run the pipeline

```bash
python run_pipeline.py --query "What is natural selection?" --config config.yaml
```

### Human-in-the-loop dashboard

```bash
streamlit run hitl/dashboard.py
```

Triggers HITL for: confidence < 0.5 | confidence 0.5-0.75 | faithfulness < 0.70 | editor removal >50%

## Data samples

The dataset comprises 5,000 texts scraped from Wikipedia, Pubmed, and ArXiv, and augmented with AI-generated 8 hallucination error types as follows:

| Error Type | Original | Hallucinated |
|------------|----------|--------------|
| Entity Replacement | "Darwin and Wallace discovered natural selection" | "Lyell and Wallace discovered natural selection" |
| Numerical Distortion | "43% of the population" | "28% of the population" |
| Negation Flip | "The enzyme catalyzes the reaction" | "The enzyme does not catalyze" |
| Temporal Confusion | "Phosphorylation precedes ubiquitination" | "Ubiquitination precedes phosphorylation" |
| Causal Reversal | "Calcium triggers contraction" | "Contraction triggers calcium" |
| Plausible Fabrication | "Binds to active site" | "Binds to active site (Chen et al., 2003)" |
| Oversimplification | "May contribute in some individuals" | "Causes disease in all individuals" |
| Citation Hallucination | "The pathway is conserved" | "The pathway is conserved (Chen et al., 2012, JMB)" |

## Run tests

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation on 2,000 test samples
python evaluation/get_metrics.py --config config.yaml --output results/scores.json
```

Metrics reported: mean RAGAS faithfulness, Pass@0.7 rate, HITL trigger rate per configuration

Expected output (heterogeneous best):
- Faithfulness: 0.772
- Pass@0.7: 86.1%
- Total HITL: 17.8%
