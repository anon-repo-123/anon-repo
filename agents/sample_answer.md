**Gatekeeper example answer**

```
    test_query = "What is natural selection?"
    test_chunks = [
        "Natural selection is the differential survival and reproduction of individuals due to differences in phenotype.",
        "Charles Darwin popularized the term 'natural selection' in his 1859 book On the Origin of Species.",
        "Natural selection acts on the heritable traits of organisms."
    ]


> Qwen-based Gatekeeper:

Should stop: False
Evaluation: {
  "confidence": 0.9,
  "reason": "Extracted from response",
  "knowledge_gaps": []
}
```

**Verifier example answer**

```
    test_answer = "Natural selection is the differential survival and reproduction of individuals due to differences in phenotype."
    test_passages = [
        "Natural selection is the differential survival and reproduction of individuals due to differences in phenotype.",
        "Charles Darwin popularized the term 'natural selection' in his 1859 book On the Origin of Species."
    ]


> Llama-based Verifier:

Is faithful: True
Evaluation: {
  "faithfulness": 0.85,
  "reason": "Answer is directly stated in the first passage, and the second passage provides additional context about the origin of the term 'natural selection'.",
  "unsupported_claims": []
}
```

**Editor example answer**

```
    test_answer = "Natural selection is, I think, the differential survival and reproduction of individuals due to differences in phenotype. It was, perhaps, first described by Charles Darwin. This is a very important concept in evolutionary biology. The mechanism acts on heritable traits over generations."


> Mistral-based Editor:

Original: Natural selection is, I think, the differential survival and reproduction of individuals due to differences in phenotype. It was, perhaps, first described by Charles Darwin. This is a very important concept in evolutionary biology. The mechanism acts on heritable traits over generations.
Edited: Natural selection is the differential survival and reproduction of individuals due to phenotypic differences, first described by Charles Darwin. It acts on heritable traits over generations, driving evolution.
Metadata: {
  "original_length": 288,
  "new_length": 209,
  "removal_percentage": 0.2743055555555556,
  "was_compressed": true,
  "exceeds_removal_threshold": false
}
```
