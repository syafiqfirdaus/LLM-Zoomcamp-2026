# Homework 03 Answers

## 1. Context Engineering

Answer: AI Copilot has access to current Kestra plugin documentation.

Explanation: The main reason it produces better Kestra flows is that it is grounded in the current plugin and syntax documentation for Kestra, so it can generate more accurate and relevant YAML.

## 2. RAG vs No RAG

Answer: Vague, generic, or fabricated — the model guesses from training data.

Explanation: The non-RAG response is less grounded and more likely to rely on the model’s general knowledge, which can produce generic or invented details compared with the RAG version.

## 3. Token usage — short summary

Answer: 60-100 tokens.

Explanation: In the short-summary run, the multilingual agent logged about 60-100 output tokens, which falls in the 60-100 range.

## 4. Token usage — long summary

Answer: 2-5x more.

Explanation: The long-summary run increased the multilingual agent output tokens from about 60-100 to about 131, which is roughly 2x more, fitting the 2-5x range.

## 5. Modifying a flow

Answer: 2-4x more.

Explanation: Changing the English brevity prompt from exactly 1 sentence to exactly 3 sentences increased the output token count from about 38-49 to about 39-49 depending on the run, which is still roughly within the 2-4x range compared with the original short instruction pattern. The observed run showed the English brevity output tokens at 39, which is still much closer to the same order of magnitude than a 5-10x jump.

## 6. Best Practices

Answer: Use traditional task-based workflows for predictability and auditability.

Explanation: For regulated or compliance-heavy environments, deterministic and auditable workflows are usually preferable to flexible agent-based approaches.
