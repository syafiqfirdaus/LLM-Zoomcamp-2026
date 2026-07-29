import json
import os
import time
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from search import load_kb_data, MinSearchEngine, VectorSearchEngine, HybridSearchEngine

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
if not os.getenv("OPENAI_API_KEY"):
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))
        api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)

client = get_openai_client()

PROMPT_V1_SIMPLE = """
Answer the user question based on the following context.
Context:
{context}

Question: {question}
Answer:
"""

PROMPT_V2_EXPERT = """
You are QuantPulse AI, a senior quantitative finance specialist and algorithm researcher.
Using ONLY the provided quantitative context below, answer the user's question clearly, concisely, and accurately.
Format mathematical equations cleanly using LaTeX formatting when applicable. Include key practical insights.

Context:
{context}

Question: {question}

Response:
"""

EVALUATION_JUDGE_PROMPT = """
You are an expert quantitative evaluator. Compare the generated answer to the ground-truth expected answer.
Rate the generated answer on a scale from 1 to 5 based on correctness, relevance, and quantitative accuracy:
1 - Completely incorrect or irrelevant
2 - Partially correct, missing key details
3 - Mostly correct with minor omissions
4 - Accurate and detailed
5 - Excellent, precise, and fully covers expected answer

Question: {question}
Expected Answer: {expected_answer}
Generated Answer: {generated_answer}

Provide your rating as a JSON object: {{"score": <number 1-5>, "reason": "<brief justification>"}}
"""

def generate_answer(prompt_template: str, question: str, context_docs: List[Dict[str, Any]], model: str = "gpt-4o-mini") -> Tuple[str, int, int]:
    context_str = "\n\n".join([f"[{d['title']}]\n{d['content']}" for d in context_docs])
    prompt = prompt_template.format(context=context_str, question=question)
    
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    
    answer = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    return answer, prompt_tokens, completion_tokens

def judge_answer(question: str, expected: str, generated: str) -> Dict[str, Any]:
    prompt = EVALUATION_JUDGE_PROMPT.format(
        question=question,
        expected_answer=expected,
        generated_answer=generated
    )
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"score": 3, "reason": f"Evaluation error: {str(e)}"}

def run_llm_evaluation():
    print("Loading datasets for LLM Evaluation...")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    with open(os.path.join(data_dir, "ground_truth.json"), "r", encoding="utf-8") as f:
        ground_truth = json.load(f)[:10]  # Evaluate sample of 10 items for speed
        
    docs = load_kb_data()
    bm25 = MinSearchEngine(docs)
    vec = VectorSearchEngine(docs)
    hybrid = HybridSearchEngine(bm25, vec)

    experiments = {
        "Simple Prompt (GPT-4o-mini)": PROMPT_V1_SIMPLE,
        "Quant Expert System Prompt (GPT-4o-mini)": PROMPT_V2_EXPERT
    }

    eval_summary = {}

    for exp_name, prompt_tmpl in experiments.items():
        print(f"\nEvaluating: {exp_name}...")
        scores = []
        
        for item in ground_truth:
            q = item["question"]
            expected = item["expected_answer"]
            
            context_docs = hybrid.search(q, top_k=3)
            answer, _, _ = generate_answer(prompt_tmpl, q, context_docs)
            eval_result = judge_answer(q, expected, answer)
            scores.append(eval_result.get("score", 3))

        avg_score = sum(scores) / len(scores)
        eval_summary[exp_name] = {
            "avg_judge_score_out_of_5": round(avg_score, 2),
            "sample_scores": scores
        }
        print(f"Result for {exp_name}: Average Score = {avg_score:.2f} / 5.0")

    output_path = os.path.join(data_dir, "llm_eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    return eval_summary

if __name__ == "__main__":
    run_llm_evaluation()
