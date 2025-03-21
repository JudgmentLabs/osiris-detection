SYSTEM_PROMPT = """You are a hallucination detector. You MUST determine if the provided answer contains hallucination or not for the question given based on the given context. An answer to a question is an hallucination if the answer is not supported by the context provided or contradicts information in the context. The answer you provided MUST be "PASS" or "FAIL"."""

HALLUCINATION_PROMPT = """Given the following question and evidence text, generate a hallucinated answer that is subtly different from the original answer. The hallucinated answer should maintain the same information but with slight variations.

Question: {question}
Evidence Text: {evidence_text}
Original Answer: {answer}

Generate a hallucinated answer and explain your reasoning for the changes made."""

VERIFICATION_PROMPT = """Given the following question, evidence text, and answer, determine if the answer contains any hallucination. An answer is considered hallucinated if it is not supported by the evidence text or contradicts information in the evidence text.

Question: {question}
Evidence Text: {evidence_text}
Answer: {answer}

Determine if the answer is hallucinated and explain your reasoning."""
