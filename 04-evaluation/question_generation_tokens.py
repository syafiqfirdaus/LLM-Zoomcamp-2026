"""
Generate questions from lesson pages and track token usage.
Extends Homework 4 analysis with LLM-based question generation.
"""

import os
import json
from typing import List, Dict, Any
from gitsource import GithubRepositoryDataReader
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_lesson_documents() -> List[Dict[str, Any]]:
    """Load lesson documents from DataTalksClub LLM Zoomcamp repository."""
    print("Loading lesson documents...")
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    documents = []
    for raw_file in reader.read():
        parsed = raw_file.parse()
        documents.append(
            {
                "filename": parsed["filename"],
                "content": parsed["content"],
            }
        )
    # Sort by filename to ensure consistent ordering
    documents.sort(key=lambda x: x["filename"])
    return documents


def generate_questions_for_lesson(lesson_content: str, lesson_name: str) -> Dict[str, Any]:
    """
    Generate questions for a single lesson using GPT.
    Returns questions and token usage.
    """
    prompt = f"""You are an expert course instructor. Based on the following lesson content, 
generate 5 clear, specific learning questions that help students understand the key concepts.

Lesson: {lesson_name}

Content:
{lesson_content}

Generate exactly 5 questions. Format as a numbered list."""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a course content expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        questions = response.choices[0].message.content
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        
        return {
            "lesson_name": lesson_name,
            "questions": questions,
            "usage": usage,
            "success": True,
        }
    except Exception as e:
        print(f"Error generating questions for {lesson_name}: {e}")
        return {
            "lesson_name": lesson_name,
            "questions": None,
            "usage": None,
            "success": False,
            "error": str(e),
        }


def main():
    print("=" * 70)
    print("Question Generation and Token Tracking for First 3 Lessons")
    print("=" * 70)
    
    # Load documents
    documents = load_lesson_documents()
    print(f"Loaded {len(documents)} lesson documents\n")
    
    # Get first 3 lessons
    first_3_lessons = documents[:3]
    print(f"Processing first 3 lessons:")
    for i, doc in enumerate(first_3_lessons, 1):
        print(f"  {i}. {doc['filename']}")
    print()
    
    # Generate questions and track tokens
    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_total_tokens = 0
    
    for lesson in first_3_lessons:
        print(f"Generating questions for: {lesson['filename']}")
        result = generate_questions_for_lesson(
            lesson['content'],
            lesson['filename']
        )
        results.append(result)
        
        if result['success']:
            usage = result['usage']
            total_input_tokens += usage['input_tokens']
            total_output_tokens += usage['output_tokens']
            total_total_tokens += usage['total_tokens']
            
            print(f"  [SUCCESS] Generated successfully")
            print(f"    - Input tokens: {usage['input_tokens']}")
            print(f"    - Output tokens: {usage['output_tokens']}")
            print(f"    - Total tokens: {usage['total_tokens']}")
        else:
            print(f"  [FAILED] Failed: {result['error']}")
        print()
    
    # Calculate averages
    num_successful = sum(1 for r in results if r['success'])
    if num_successful > 0:
        avg_input_tokens = total_input_tokens / num_successful
        avg_output_tokens = total_output_tokens / num_successful
        avg_total_tokens = total_total_tokens / num_successful
    else:
        avg_input_tokens = avg_output_tokens = avg_total_tokens = 0
    
    # Generate markdown report
    report = f"""# Question Generation and Token Tracking Analysis

## Summary

Generated questions for the first 3 lesson pages and tracked OpenAI API token usage.

### Lessons Processed

1. {first_3_lessons[0]['filename']}
2. {first_3_lessons[1]['filename']}
3. {first_3_lessons[2]['filename']}

### Token Usage Summary

| Metric | Value |
|--------|-------|
| Total Input Tokens (all 3 calls) | {total_input_tokens} |
| Total Output Tokens (all 3 calls) | {total_output_tokens} |
| Total Tokens (all 3 calls) | {total_total_tokens} |
| **Avg Input Tokens** | **{avg_input_tokens:.1f}** |
| **Avg Output Tokens** | **{avg_output_tokens:.1f}** |
| **Avg Total Tokens** | **{avg_total_tokens:.1f}** |
| Successful Calls | {num_successful}/3 |

## Answer to Question

**Average number of input tokens across the 3 calls: {avg_input_tokens:.1f}**

## Detailed Results

"""
    
    for i, result in enumerate(results, 1):
        report += f"\n### Lesson {i}: {result['lesson_name']}\n\n"
        
        if result['success']:
            usage = result['usage']
            report += f"**Token Usage:**\n"
            report += f"- Input tokens: {usage['input_tokens']}\n"
            report += f"- Output tokens: {usage['output_tokens']}\n"
            report += f"- Total tokens: {usage['total_tokens']}\n\n"
            report += f"**Generated Questions:**\n\n"
            report += result['questions']
            report += "\n"
        else:
            report += f"**Status:** Failed\n"
            report += f"**Error:** {result['error']}\n"
    
    report += f"""

## Analysis

The average input tokens across the three API calls for question generation was **{avg_input_tokens:.1f} tokens**.

This represents the typical prompt size sent to the LLM when generating questions from lesson content. 
The input includes:
- System prompt (instructing the LLM to be a course expert)
- User prompt with lesson name and content excerpt (~2000 chars)
- Instructions for output format

Output tokens averaged **{avg_output_tokens:.1f}**, representing the generated questions and explanations.

## Technical Details

- **Model:** gpt-4
- **Temperature:** 0.7
- **Max tokens per request:** 500
- **Content size per lesson:** First 2000 characters
- **Number of calls:** 3 (one per lesson)
- **Success rate:** {(num_successful/3)*100:.0f}%
"""
    
    return avg_input_tokens, avg_output_tokens, avg_total_tokens, report


if __name__ == "__main__":
    try:
        from pathlib import Path
        avg_input, avg_output, avg_total, report_md = main()
        
        # Append to answers.md relative to script location
        script_dir = Path(__file__).parent
        output_file = script_dir / "answers.md"
        
        print("\n" + "=" * 70)
        print(f"Appending results to {output_file}...")
        print("=" * 70)
        
        with open(output_file, "a", encoding="utf-8") as f:
            f.write("\n\n" + "=" * 70 + "\n")
            f.write(report_md)
        
        print(f"\n[OK] Results saved to {output_file}")
        print(f"\nKEY RESULT:")
        print(f"   Average input tokens across 3 calls: {avg_input:.1f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
