import torch
import argparse
import jsonlines
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from tqdm import tqdm

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load local jsonlines file
    with jsonlines.open(args.input_file) as reader:
        data = [line for line in reader]

    # Convert list of dictionaries to dictionary of lists
    data_dict = {key: [d[key] for d in data] for key in data[0]}
    dataset = Dataset.from_dict(data_dict)

    # Check how many lines have already been written to the output file
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            existing_lines = sum(1 for _ in f)
        print(f"Skipping {existing_lines} already processed lines.")
    else:
        existing_lines = 0

    # Skip already processed lines
    if existing_lines > 0:
        dataset = dataset.select(range(existing_lines, len(dataset)))

    def compute_scores(batch):
        inputs = tokenizer(batch[args.text_column], return_tensors="pt", padding="longest", truncation=True, max_length=args.max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1).float().cpu().numpy()

        batch["score"] = logits.tolist()
        batch["int_score"] = [int(round(max(0, min(score, 5)))) for score in logits]
        return batch

    # Process and write each batch incrementally
    with jsonlines.open(args.output_file, mode='a') as writer:
        for batch in tqdm(dataset.iter(batch_size=args.batch_size), total=(len(dataset) + args.batch_size - 1) // args.batch_size):
            processed_batch = compute_scores(batch)
            writer.write_all([dict(zip(batch.keys(), vals)) for vals in zip(*processed_batch.values())])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="north/scandinavian_education_classifier_bert")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input jsonlines file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output jsonlines file")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for processing")

    args = parser.parse_args()
    main(args)
