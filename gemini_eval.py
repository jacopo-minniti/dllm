import json
import re
import time
import os
import random
import concurrent.futures
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv

load_dotenv()

# =====================================================================
# CONFIGURATION
# =====================================================================
DATA_PATH = ".models/LLaDA-8B-Base/tulu-3-sft-mixture/base-puma-bptt-cab/lr5e-05_bs256_puma-th0.15_bptt4_cab-b512-e1024-rl15/checkpoint-final/evals/mnt512_s256_bs32_t0.0_th0.15_loop_nf4/.models__LLaDA-8B-Base__tulu-3-sft-mixture__base-puma-bptt-cab__lr5e-05_bs256_puma-th0.15_bptt4_cab-b512-e1024-rl15__checkpoint-final/samples_math500_reasoning_2026-04-11T21-23-25.222241.jsonl"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_ID = "gemini-3.1-flash-lite-preview"

GEMINI_SYSTEM_PROMPT = """You are an expert mathematics evaluator. Your task is to compare a 'Gold Answer' to a 'Student Response' and determine if the student's final answer is mathematically equivalent to the gold answer.

CRITICAL GUIDELINES:
1. Focus ONLY on the final answer provided by the student. Ignore any logical flaws, hallucinations, or errors in their reasoning steps leading up to it. 
2. Evaluate based on strict mathematical equivalence. For example: 
   - 1/2 is equivalent to 2/4, 0.5, and \\frac{1}{2}.
   - \\sqrt{51} is equivalent to 51^{0.5}.
   - x=5 is equivalent to 5.
3. Ignore minor formatting differences, LaTeX variations (e.g., \\text{cm} vs cm), or extra surrounding text, as long as the mathematical core is identical.
4. First, briefly reason about what the student's final answer is and whether it matches the gold answer.
5. You MUST conclude your response with EXACTLY \\boxed{True} (if they are equivalent) or \\boxed{False} (if they are not). Do not use any other formatting for the final boolean.
"""
# =====================================================================

def extract_boolean_from_response(text: str) -> bool:
    """Parses the \boxed{True} or \boxed{False} from Gemini's response."""
    match = re.search(r'\\boxed{\s*(True|False)\s*}', text, re.IGNORECASE)
    if match:
        result = match.group(1).lower()
        return result == 'true'
    
    # Fallback in case the model forgets the box but says True/False at the very end
    if "true" in text.lower()[-20:]:
        return True
    return False

def evaluate_sample(client: genai.Client, gold_answer: str, model_response: str, max_retries=5) -> tuple[bool, str]:
    """Calls Gemini to evaluate a single sample with exponential backoff."""
    prompt = f"""
GOLD ANSWER:
{gold_answer}

STUDENT RESPONSE:
{model_response}

Is the student's final answer mathematically equivalent to the gold answer?
"""
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=GEMINI_SYSTEM_PROMPT,
                    temperature=0.0, 
                )
            )
            
            evaluation_text = response.text
            is_correct = extract_boolean_from_response(evaluation_text)
            return is_correct, evaluation_text
            
        except APIError as e:
            # Exponential backoff with jitter to prevent thundering herd when 5 threads fail at once
            sleep_time = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(sleep_time)
        except Exception as e:
            return False, f"Unexpected Error: {str(e)}"
            
    return False, "Failed after max retries."

def worker(task_data: dict) -> tuple[int, bool, str]:
    """Worker function to be executed by each thread."""
    client = task_data['client']
    line_num = task_data['line_num']
    gold_answer = task_data['gold_answer']
    model_response = task_data['model_response']
    
    if not model_response:
        return line_num, False, f"Sample {line_num} | Correct: False (No model response found)"
        
    is_correct, _ = evaluate_sample(client, gold_answer, model_response)
    
    # Optional: We add a tiny sleep here to avoid instantly maxing out RPM limits on lower tiers
    time.sleep(0.2) 
    
    return line_num, is_correct, f"Sample {line_num} | Correct: {is_correct}"

def main():
    # Setup paths
    base_dir = os.path.dirname(DATA_PATH) or '.'
    log_file_path = os.path.join(base_dir, "gemini_eval.txt")
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Read all tasks into memory
    tasks = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            data = json.loads(line)
            gold_answer = data.get('doc', {}).get('answer', '')
            
            filtered_resps = data.get('filtered_resps', [])
            model_response = filtered_resps[0] if filtered_resps else ""
            
            tasks.append({
                'client': client,
                'line_num': line_num,
                'gold_answer': gold_answer,
                'model_response': model_response
            })

    total_samples = len(tasks)
    correct_samples = 0

    # Custom logger to print to console AND write to file
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        def log_msg(msg: str):
            print(msg)
            log_file.write(msg + '\n')
            log_file.flush() # Ensure it writes to disk immediately

        log_msg(f"Starting parallel evaluation of {DATA_PATH}...")
        log_msg(f"Using {GEMINI_MODEL_ID} with 5 worker threads.")
        log_msg(f"Logging output to {log_file_path}\n")

        if total_samples == 0:
            log_msg("No valid samples found in the dataset.")
            return

        # Execute threads in parallel using ThreadPoolExecutor
        # 5 workers is generally a safe sweet spot for API I/O without triggering massive rate limits
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(worker, task): task for task in tasks}
            
            # Process as they complete (Note: output will print out of order, which is normal for async)
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    line_num, is_correct, result_msg = future.result()
                    if is_correct:
                        correct_samples += 1
                    log_msg(result_msg)
                except Exception as exc:
                    log_msg(f"Task generated an exception: {exc}")

        # Final Summary
        accuracy = correct_samples / total_samples
        log_msg("\n" + "="*40)
        log_msg("EVALUATION COMPLETE")
        log_msg("="*40)
        log_msg(f"Total Samples:   {total_samples}")
        log_msg(f"Correct Answers: {correct_samples}")
        log_msg(f"Accuracy Score:  {accuracy:.2%} ({accuracy:.4f})")

if __name__ == "__main__":
    main()