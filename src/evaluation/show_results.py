import re
import argparse
import json
from sklearn.metrics import recall_score, precision_score, f1_score
import pandas as pd

def show_metrics(model, output_file):
    with open(output_file, 'r') as f:
        results = [json.loads(line) for line in f]
        
    # Calculate metrics
    df = pd.DataFrame.from_records(results)
    df['is_halu'] = df['labels'].apply(lambda x: len(x)>0)

    # Check for instances where gpt_pred is a float instead of a dict
    float_instances = df[df['gpt_pred'].apply(lambda x: isinstance(x, float))].shape[0]
    if float_instances > 0:
        print(f"Warning: Found {float_instances} instances where gpt_pred is a float instead of a dict")
        # For these instances, replace float with empty dict to avoid errors
        df['gpt_pred'] = df['gpt_pred'].apply(lambda x: {} if isinstance(x, float) else x)

    df['pred_halu'] = df['gpt_pred'].apply(lambda x: len(x.get('hallucination_list', [])) > 0)

    # # Print overall metrics
    print(f"\nOverall Metrics: for {model}")
    # Calculate accuracy by counting matches and dividing by total
    num_correct = (df['is_halu'] == df['pred_halu']).sum()
    total = len(df)
    accuracy = num_correct / total
    print(f"\nAccuracy: {accuracy:.3f}")
    print(f"Recall: {recall_score(df['is_halu'], df['pred_halu']):.3f}")
    print(f"Precision: {precision_score(df['is_halu'], df['pred_halu']):.3f}")
    print(f"F1: {f1_score(df['is_halu'], df['pred_halu']):.3f}")
    
    # Print metrics by task
    for task in ['QA', 'Summary', 'Data2txt']:
        temp = df[df['task_type']==task]
        print(f"\n{task} Metrics:")
        # Calculate accuracy by counting matches and dividing by total
        num_correct = (df['is_halu'] == df['pred_halu']).sum()
        total = len(df)
        accuracy = num_correct / total
        print(f"\nAccuracy: {accuracy:.3f}")
        print(f"Recall: {recall_score(temp['is_halu'], temp['pred_halu']):.3f}")
        print(f"Precision: {precision_score(temp['is_halu'], temp['pred_halu']):.3f}")
        print(f"F1: {f1_score(temp['is_halu'], temp['pred_halu']):.3f}")

def evaluate_ragtruth(results_dir):
    ragtruth_pattern = re.compile(r"(.+)_predictions\.jsonl")
    for file in results_dir.glob("gpt-4o_raw_predictions.jsonl"):
        match = ragtruth_pattern.match(file.name)
        if match:
            model = match.group(1)
            print(f"\nEvaluating RAGTruth for model: {model}")
            show_metrics(model, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run evaluations for HaluBench and RAGTruth')
    parser.add_argument('--task', type=str, default='qa', help='Task name for evaluation (default: qa)')
    parser.add_argument('--results_dir', type=str, default='./benchmarks/ragtruth/results', help='Path to results directory')
    args = parser.parse_args()

    evaluate_ragtruth(args.results_dir)









