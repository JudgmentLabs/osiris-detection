import json
from tqdm import tqdm
from argparse import ArgumentParser
from openai import OpenAI
import os 
import torch
from transformers import pipeline

client = OpenAI()

TEMPLATES = {
    "QA": (
        "Below is a question:\n"
        "{question}\n\n"
        "Below are related passages:\n"
        "{reference}\n\n"
        "Below is an answer:\n"
        "{response}\n\n"
        "Your task is to determine whether the summary contains either or both of the following two types of hallucinations:\n"
        "1. conflict: instances where the summary presents direct contraction or opposition to the original news;\n"
        "2. baseless info: instances where the generated summary includes information which is not substantiated by or inferred from the original news. \n"
        "Then, compile the labeled hallucinated spans into a JSON dict, with a key \"hallucination_list\" and its value is a list of hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{\"hallucination_list\": [hallucination span1, hallucination span2, ...]}}. Otherwise, leave the value as a empty list as following: {{\"hallucination_list\": []}}.\n"
        "Only output the JSON dict, no other text or explanations."
    ),
    "Summary": (
        "Below is the original news:\n" 
        "{reference}\n\n"
        "Below is a summary of the news:\n"
        "{response}\n"
        "Your task is to determine whether the summary contains either or both of the following two types of hallucinations:\n"
        "1. conflict: instances where the summary presents direct contraction or opposition to the original news;\n"
        "2. baseless info: instances where the generated summary includes information which is not substantiated by or inferred from the original news. \n"
        "Then, compile the labeled hallucinated spans into a JSON dict, with a key \"hallucination_list\" and its value is a list of hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{\"hallucination_list\": [hallucination span1, hallucination span2, ...]}}. Otherwise, leave the value as a empty list as following: {{\"hallucination_list\": []}}.\n"
        "Only output the JSON dict, no other text or explanations."
    ),
    "Data2txt": (
        "Below is a structured data in the JSON format:\n"
        "{reference}\n\n"
        "Below is an overview article written in accordance with the structured data:\n"
        "{response}\n\n"
        "Your task is to determine whether the summary contains either or both of the following two types of hallucinations:\n"
        "1. conflict: instances where the summary presents direct contraction or opposition to the original news;\n"
        "2. baseless info: instances where the generated summary includes information which is not substantiated by or inferred from the original news. \n"
        "Then, compile the labeled hallucinated spans into a JSON dict, with a key \"hallucination_list\" and its value is a list of hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{\"hallucination_list\": [hallucination span1, hallucination span2, ...]}}. Otherwise, leave the value as a empty list as following: {{\"hallucination_list\": []}}.\n"
        "Only output the JSON dict, no other text or explanations."
    ),
}

def get_prompt(data):
    if data['task_type'] == 'QA':
        return TEMPLATES[data['task_type']].format(
            question=data['question'],
            reference=data['reference'],
            response=data['response']
        )
    return TEMPLATES[data['task_type']].format(
        reference=data['reference'],
        response=data['response']
    )

def process(model_path, test_file, output_file):
    """Process a chunk of test data with the given model using threading."""

    existing_questions = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if "raw_pred" in data:
                    existing_questions.add(data["id"])

    with open(test_file) as f:
        test_data = [json.loads(line) for line in f]

    print(f"Loading pipeline for: {model_path}")
    
    pipe = pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    
    print(f"Pipeline loaded successfully")


    # Process each item in the chunk
    for data in tqdm(test_data, desc=f"Inference: {output_file.split('/')[-1]}"):
        if data["id"] in existing_questions:
            continue

        prompt = get_prompt(data)
        messages = [
            {"role": "user", "content": prompt},
        ]

        try:

            if "gpt" in model_path:
                completion = client.chat.completions.create(
                    model=model_path,
                    messages=messages,
                )

                response = completion.choices[0].message.content
                    
            else:
                outputs = pipe(
                    messages,
                    max_new_tokens=256,
                )
                response = outputs[0]["generated_text"][-1]

        except Exception as e:
            print(f"Error: {e}")
            continue
      
        result = dict(data)
        result['raw_pred'] = response
        
        dump_jsonl(result, output_file, append=True)

def run_single_model(model_path, test_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
       
    output_file = f"{output_dir}/{model_path.split('/')[-1]}_raw_predictions.jsonl"
    print(f"Evaluating {model_path} at {output_file}")

    process(model_path, test_file, output_file)

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--models", nargs="+", default=[
        "judgmentlabs/Qwen2.5-Osiris-7B-Instruct",
    ], help="List of models to evaluate")
    parser.add_argument('--test_file', default="./benchmarks/ragtruth/test.jsonl", help='Path to test data')
    parser.add_argument('--output_dir', default="./benchmarks/ragtruth/results", help='Path to output directory')
    args = parser.parse_args()

    for model in args.models:
        run_single_model(model, args.test_file, args.output_dir)
