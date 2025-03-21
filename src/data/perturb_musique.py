import asyncio
import json
import os
import random
import logging

from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm import tqdm
from pathlib import Path

from prompts import HALLUCINATION_PROMPT, VERIFICATION_PROMPT, SYSTEM_PROMPT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Hallucination(BaseModel):
    hallucinated_answer: str
    reasoning: str


class Verification(BaseModel):
    is_hallucinated: bool
    reasoning: str


class DatasetPerturbator:
    def __init__(self, dataset_path, output_dir="data/perturbed", max_samples=None):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.max_samples = max_samples

    async def hallucination_query(self, sample, client, max_retries=3):
        all_paragraphs = [p["paragraph_text"] for p in sample["paragraphs"]]
        evidence_text = " ".join(all_paragraphs)

        try:
            response = await client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": HALLUCINATION_PROMPT.format(
                        question=sample["question"],
                        evidence_text=evidence_text,
                        answer=sample["answer"]
                    )}
                ],
                temperature=0.7,
                response_format=Hallucination
            )

            return {
                "question": sample["question"],
                "evidence_text": sample["paragraphs"],
                "answer": sample["answer"],
                "hallucinated_answer": response.choices[0].message.parsed.hallucinated_answer,
                "reasoning": response.choices[0].message.parsed.reasoning,
                "is_hallucinated": True
            }
        except Exception as e:
            print(
                f"Error in hallucination query for question: {sample['question']}")
            print(f"Error: {str(e)}")
            return None

    async def verification_query(self, sample, client, max_retries=3):
        all_paragraphs = [p["paragraph_text"] for p in sample["paragraphs"]]
        evidence_text = " ".join(all_paragraphs)

        try:
            response = await client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": VERIFICATION_PROMPT.format(
                        question=sample["question"],
                        evidence_text=evidence_text,
                        answer=sample["answer"]
                    )}
                ],
                temperature=0.3,
                response_format=Verification
            )

            return {
                "question": sample["question"],
                "evidence_text": sample["paragraphs"],
                "answer": sample["answer"],
                "reasoning": response.choices[0].message.parsed.reasoning,
                "is_hallucinated": response.choices[0].message.parsed.is_hallucinated
            }
        except Exception as e:
            print(
                f"Error in verification query for question: {sample['question']}")
            print(f"Error: {str(e)}")
            return None

    async def process_sample(self, sample, client):
        if random.random() < 0.5:
            return await self.hallucination_query(sample, client)
        else:
            return await self.verification_query(sample, client)

    def validate_record(self, record):
        """Validate that record has all required fields."""
        required_fields = ['question', 'answer',
                           'evidence_text', 'is_hallucinated']
        for field in required_fields:
            if field not in record:
                raise ValueError(f"Record missing required field: {field}")

        if not isinstance(record['evidence_text'], list):
            raise ValueError("evidence_text must be a list")

        if not isinstance(record['is_hallucinated'], bool):
            raise ValueError("is_hallucinated must be a boolean value")

    def format_record(self, record):
        """Format a record into the conversational format with error handling."""
        try:
            # Validate record structure
            self.validate_record(record)

            # Join evidence text
            evidence_text = []
            for evidence in record['evidence_text']:
                if isinstance(evidence, dict) and 'title' in evidence:
                    evidence_text.append(
                        f"{evidence['title']}: {evidence['paragraph_text']}")
                else:
                    evidence_text.append(evidence['paragraph_text'])
            context = "\n\n".join(evidence_text)

            # Create the formatted record
            return {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"CONTEXT: {context} \nQUESTION: {record['question']} \nANSWER: {record['answer']}:"
                    },
                    {
                        "role": "assistant",
                        "content": "PASS" if not record.get('is_hallucinated', False) else "FAIL" + f"\nREASONING: {record['reasoning']}"
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error formatting record: {str(e)}")
            logger.debug(f"Problematic record: {json.dumps(record, indent=2)}")
            return None

    async def create_dataset_async(self):
        client = AsyncOpenAI()
        dataset = []
        successful_records = 0
        skipped_records = 0
        total_records = 0

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        perturbed_file = os.path.join(
            self.output_dir, Path(self.dataset_path).name)
        conversational_file = os.path.join(self.output_dir, Path(
            self.dataset_path).stem + "_conversational.jsonl")

        # First create perturbed dataset
        with open(self.dataset_path, 'r') as f, open(perturbed_file, 'w') as out_f:
            for line in tqdm(f, desc="Perturbing dataset"):
                if not line.strip():  # Skip empty lines
                    continue

                total_records += 1
                sample = json.loads(line)
                result = await self.process_sample(sample, client)

                if result:
                    json.dump(result, out_f)
                    out_f.write('\n')
                    dataset.append(result)
                    successful_records += 1
                else:
                    skipped_records += 1

                if self.max_samples and len(dataset) >= self.max_samples:
                    break

                # Add a small delay to avoid rate limiting
                await asyncio.sleep(0.1)

        logger.info(f"Perturbed dataset saved to {perturbed_file}")
        logger.info(f"Successfully perturbed {successful_records} records")
        logger.info(
            f"Skipped {skipped_records} records out of {total_records} total")

        # Now convert to conversational format
        successful_conversions = 0
        skipped_conversions = 0

        with open(perturbed_file, 'r') as f, open(conversational_file, 'w') as out_f:
            for line in tqdm(f, desc="Converting to conversational format"):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                    formatted_record = self.format_record(record)
                    if formatted_record:
                        json.dump(formatted_record, out_f)
                        out_f.write('\n')
                        successful_conversions += 1
                    else:
                        skipped_conversions += 1
                except Exception as e:
                    logger.error(f"Error converting record: {str(e)}")
                    skipped_conversions += 1

        logger.info(f"Conversational dataset saved to {conversational_file}")
        logger.info(f"Successfully converted {successful_conversions} records")
        logger.info(f"Skipped {skipped_conversions} records")

        return dataset

    def perturb(self):
        return asyncio.run(self.create_dataset_async())


if __name__ == "__main__":
    perturbator = DatasetPerturbator(
        dataset_path="datasets/musique_full_v1.0_train.jsonl",
        output_dir="datasets/perturbed",
        max_samples=5
    )
    perturbator.perturb()
