<div align="center">
  <h1>Osiris: Lightweight Hallucination Evaluation Model</h1>
</div>

## 🔥 News

- arxiv is coming out soon.
- Models checkpoints are available on [huggingface](https://huggingface.co/collections/judgmentlabs/osiris-detection-67dca1fed2ebad58fd8475ff):
    - Qwen2.5-Osiris-0.5B-Instruct: [`judgmentlabs/Qwen2.5-Osiris-0.5B-Instruct`](https://huggingface.co/judgmentlabs/Qwen2.5-Osiris-0.5B-Instruct)
    - Qwen2.5-Osiris-1.5B-Instruct: [`judgmentlabs/Qwen2.5-Osiris-1.5B-Instruct`](https://huggingface.co/judgmentlabs/Qwen2.5-Osiris-1.5B-Instruct)
    - Qwen2.5-Osiris-3B-Instruct: [`judgmentlabs/Qwen2.5-Osiris-3B-Instruct`](https://huggingface.co/judgmentlabs/Qwen2.5-Osiris-3B-Instruct)
    - Qwen2.5-Osiris-7B-Instruct: [`judgmentlabs/Qwen2.5-Osiris-7B-Instruct`](https://huggingface.co/judgmentlabs/Qwen2.5-Osiris-7B-Instruct)

- 📊 Dataset is available at [here](https://huggingface.co/datasets/judgmentlabs/osiris-musique-v1.0).

## Usage

```python
from src.data.perturb_musique import DatasetPerturbator

perturbator = DatasetPerturbator(
    dataset_path="/path/to/your/dataset.jsonl",
    output_dir="/path/to/save/perturbed/dataset"
)

perturbator.perturb()
```

## Evaluation

```python
# Navigate to the evaluation directory
cd src/data/evaluation

# Run the RAGTruth benchmark on models defined in this script
# This evaluates how well models detect hallucinations in RAG contexts
bash ragtruth_predict.sh

# Format the RAGTruth benchmark results into structured JSON files if necessary
# This prepares the data for analysis and visualization
bash format_predictions.sh

# Calculate and display evaluation metrics
# Shows Recall, Precision, and F1 scores for hallucination detection
python show_results.py
```




## Training

We used [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for efficient fine-tuning. A sample configuration can be found [here](https://qwen.readthedocs.io/en/latest/training/SFT/llama_factory.html).

