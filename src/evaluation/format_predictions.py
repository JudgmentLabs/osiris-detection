import os
import json
import glob
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def process_raw_predictions(results_dir, max_workers=16):
    """
    Process all *_raw_predictions.jsonl files in the specified directory,
    send each raw prediction to GPT API, and save results to new files.
    Uses ThreadPoolExecutor to process one file per thread.
    """

    raw_files = glob.glob(os.path.join(results_dir, "*_raw_predictions.jsonl"))
    print(f"Found {len(raw_files)} raw prediction files to process")
    
    for raw_file in raw_files:
        # Determine output filename
        base_name = os.path.basename(raw_file).replace("_raw_predictions.jsonl", "")
        output_file = os.path.join(results_dir, f"{base_name}_gpt_predictions.jsonl")

        process_file(raw_file, output_file)
    
def process_file(input_file, output_file):
    """Process a single raw predictions file through the GPT API."""
    # Read input data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    existing_data = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                existing_data.add(entry['id'])
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        # Submit all tasks
        future_to_entry = {
            executor.submit(process_entry, entry): entry
            for entry in data
            if 'raw_pred' in entry and entry['id'] not in existing_data
        }

        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_entry),
                           total=len(future_to_entry),
                           desc=f"Processing {os.path.basename(input_file)}",
                           leave=False):
            entry = future_to_entry[future]
            try:
                result = future.result()
                if result:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result) + '\n')
            except Exception as e:
                print(f"Error processing entry {entry['id']}: {e}")
    
    return input_file

def process_entry(entry):
    """Process a single entry."""
    
    try:
        gpt_response = call_gpt_api(entry['raw_pred'])
        entry['gpt_pred'] = json.loads(gpt_response)
        return entry
    except Exception as e:
        print(f"Error in processing entry {entry['id']}: {e}")
        return None

def call_gpt_api(text):
    """Call the GPT API with the given text and return the response."""
    # Make sure your API key is set in the environment variable OPENAI_API_KEY
    # or configure it here: openai.api_key = "your-api-key"

    SYSTEM_PROMPT = """You are a JSON repair tool. Output only valid JSON, no explanations.
# Common errors you fix:
# 1. Missing commas between array items: ["item1" "item2"] → ["item1", "item2"]
# 2. Unclosed brackets: {"list": [{"item": "value"} → {"list": [{"item": "value"}]}
# 3. Missing quotes: {list: [value]} → {"list": ["value"]}
# 4. Trailing commas: ["item1", "item2",] → ["item1", "item2"]
# 5. Unstructured lists: "Hallucinations: 1. First item 2. Second item" → {"hallucination_list": ["First item", "Second item"]}
# 6. Bullet points: "• First item • Second item" → {"hallucination_list": ["First item", "Second item"]}
# 7. Numbered lists: "1) First item 2) Second item" → {"hallucination_list": ["First item", "Second item"]}
# 8. Line-separated items: "First item\nSecond item" → {"hallucination_list": ["First item", "Second item"]}

# If you see a plain text response with phrases like "I found these hallucinations:" or "Hallucinated content:",
# extract the listed items and format them as a proper JSON array in the hallucination_list.


# The JSON must follow this exact format:
# {"hallucination_list": ["span1", "span2"]}
# or for no hallucinations: {"hallucination_list": []}

# If you see {"type": "conflict", "span": "text"} format, extract ONLY the span value.
# Example:
# Input: {"hallucination_list": [{"type": "conflict", "span": "text1"}, {"type": "baseless", "span": "text2"}]}
# Output: {"hallucination_list": ["text1", "text2"]}"""

    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT}, 
            {"role": "user", "content": text}],
        max_tokens=1024,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process raw predictions and format them.")
    parser.add_argument("--results_dir", type=str, default="./benchmarks/ragtruth/results",
                        help="Directory containing raw prediction files")
    
    args = parser.parse_args()
    process_raw_predictions(args.results_dir)