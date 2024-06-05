import os
import argparse
import json
import time
import jsonlines
import logging
from tqdm import tqdm
import requests
from google.auth import default
from google.auth.transport.requests import Request
from google.cloud import bigquery

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the templates from the template.json file
with open('template.json', 'r') as f:
    templates = json.load(f)

# Define the model configuration
generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}
safety_settings = []

def prepare_input_data(json_lines_file, dataset_name, table_name, language, batch_size=500):
    client = bigquery.Client()
    dataset_ref = client.dataset(dataset_name)

    # Create the dataset if it doesn't exist
    try:
        client.get_dataset(dataset_ref)
        logging.info(f"Dataset {dataset_name} already exists.")
    except Exception:
        client.create_dataset(bigquery.Dataset(dataset_ref))
        logging.info(f"Created dataset {dataset_name}")

    table_ref = dataset_ref.table(table_name)
    schema = [
        bigquery.SchemaField("request", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("status", "STRING"),
        bigquery.SchemaField("response", "STRING"),
    ]

    # Create the table if it doesn't exist
    try:
        client.get_table(table_ref)
        logging.info(f"Table {table_name} already exists.")
    except Exception:
        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table)
        logging.info(f"Created table {table_name}")

    chat_prompt = templates[language]

    with jsonlines.open(json_lines_file, mode='r') as reader:
        lines = list(reader)

    total_lines = len(lines)
    rows_to_insert = []

    with tqdm(total=total_lines, desc="Preparing input data") as pbar:
        for line in lines:
            input_text = line.get("text", line.get("content"))
            if input_text:
                prompt = chat_prompt.format(content=input_text)
                rows_to_insert.append({"request": json.dumps({"contents": [{"role": "user", "parts": [{"text": prompt}]}]})})
            else:
                logging.error(f"Field 'text' or 'content' not found in line: {line}")
            pbar.update(1)

    total_batches = len(rows_to_insert) // batch_size + (1 if len(rows_to_insert) % batch_size != 0 else 0)
    
    with tqdm(total=total_batches, desc="Inserting rows into BigQuery") as pbar:
        for i in range(0, len(rows_to_insert), batch_size):
            batch = rows_to_insert[i:i + batch_size]
            errors = client.insert_rows_json(table_ref, batch)
            if errors:
                logging.error(f"Errors occurred while inserting rows: {errors}")
            pbar.update(1)

    logging.info(f"Inserted {len(rows_to_insert)} rows into {table_name}")

def submit_batch_prediction_job(project_id, location, model_id, job_name, dataset_name, input_table_name, output_table_name):
    ENDPOINT = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/batchPredictionJobs"
    
    credentials, _ = default()
    credentials.refresh(Request())
    auth_token = credentials.token

    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "name": job_name,
        "displayName": job_name,
        "model": f"projects/{project_id}/locations/{location}/publishers/google/models/{model_id}",
        "inputConfig": {
            "instancesFormat": "bigquery",
            "bigquerySource": {
                "inputUri": f"bq://{project_id}.{dataset_name}.{input_table_name}"
            }
        },
        "outputConfig": {
            "predictionsFormat": "bigquery",
            "bigqueryDestination": {
                "outputUri": f"bq://{project_id}.{dataset_name}.{output_table_name}"
            }
        }
    }

    response = requests.post(ENDPOINT, headers=headers, json=payload)
    if response.status_code == 200:
        logging.info(f"Batch prediction job submitted successfully. Job ID: {response.json()['name']}")
    else:
        logging.error(f"HTTPError: {response.status_code} {response.reason}")
        logging.error(f"Response content: {response.content}")
        response.raise_for_status()

def main():
    parser = argparse.ArgumentParser(description="Submit a batch prediction job to Vertex AI.")
    parser.add_argument('--project_id', type=str, required=True, help='Google Cloud project ID.')
    parser.add_argument('--dataset_name', type=str, required=True, help='BigQuery dataset name.')
    parser.add_argument('--input_table_name', type=str, required=True, help='BigQuery input table name.')
    parser.add_argument('--output_table_name', type=str, required=True, help='BigQuery output table name.')
    parser.add_argument('--job_name', type=str, required=True, help='Name of the batch prediction job.')
    parser.add_argument('--input_jsonl_file', type=str, required=True, help='Path to the input JSONL file.')
    parser.add_argument('--language', type=str, choices=['en', 'sv', 'da', 'nb', 'nn'], default='en', help='Language for the prompt (default: en).')
    parser.add_argument('--model_id', type=str, required=True, help='ID of the model to use for prediction.')
    parser.add_argument('--location', type=str, required=True, help='Location of the Vertex AI endpoint.')
    parser.add_argument('--dryrun', action='store_true', help='Perform a dry run, showing an example of the formatted input.')

    args = parser.parse_args()

    if args.dryrun:
        with jsonlines.open(args.input_jsonl_file, mode='r') as reader:
            for line in reader:
                input_text = line.get("text", line.get("content"))
                if input_text:
                    chat_prompt = templates[args.language]
                    prompt = chat_prompt.format(content=input_text)
                    example = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
                    logging.info(json.dumps(example, indent=4))
                    break
        return

    prepare_input_data(args.input_jsonl_file, args.dataset_name, args.input_table_name, args.language)
    submit_batch_prediction_job(args.project_id, args.location, args.model_id, args.job_name, args.dataset_name, args.input_table_name, args.output_table_name)

if __name__ == "__main__":
    main()
