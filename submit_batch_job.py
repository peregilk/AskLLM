import os
import argparse
import json
import time
import jsonlines
import logging
import requests
from google.auth import default
from google.auth.transport.requests import Request
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Authentication
def get_auth_token():
    credentials, _ = default()
    credentials.refresh(Request())
    return credentials.token

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
# Removed the problematic safety settings
safety_settings = []

# Prepare input data and insert into BigQuery
def prepare_input_data(project_id, dataset_name, table_name, input_jsonl_file, chat_prompt, dryrun=False):
    client = bigquery.Client(project=project_id)
    dataset_ref = client.dataset(dataset_name)
    table_ref = dataset_ref.table(table_name)

    # Create table if it doesn't exist
    try:
        table = client.get_table(table_ref)
        logging.info(f"Table {table_name} already exists.")
    except NotFound:
        schema = [bigquery.SchemaField("request", "STRING")]
        table = bigquery.Table(table_ref, schema=schema)
        table = client.create_table(table)
        logging.info(f"Created table {table_name}")

    # Load input data from JSONL file into BigQuery
    with open(input_jsonl_file, 'r') as f:
        rows_to_insert = []
        for line in f:
            input_data = json.loads(line.strip())
            input_text = input_data.get("text") or input_data.get("content")
            if not input_text:
                logging.error(f"No text or content field found in line: {line.strip()}")
                continue
            prompt = chat_prompt.format(content=input_text)
            formatted_request = {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": prompt}]
                }],
                "generation_config": generation_config,
                "safety_settings": safety_settings
            }
            if dryrun:
                logging.info(json.dumps(formatted_request, indent=4))
                return
            rows_to_insert.append({"request": json.dumps(formatted_request)})

    errors = client.insert_rows_json(table, rows_to_insert)
    if errors:
        logging.error(f"Failed to insert rows: {errors}")
        raise Exception(f"Failed to insert rows: {errors}")
    else:
        logging.info(f"Inserted {len(rows_to_insert)} rows into {table_name}")

# Submit batch prediction job
def submit_batch_prediction_job(project_id, location, model_id, job_name, dataset_name, input_table_name, output_table_name):
    headers = {
        "Authorization": f"Bearer {get_auth_token()}",
        "Content-Type": "application/json"
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
    
    response = requests.post(f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/batchPredictionJobs", headers=headers, json=payload)
    response.raise_for_status()
    job_id = response.json()['name']
    logging.info(f"Batch prediction job submitted successfully. Job ID: {job_id}")
    return job_id

# Main function
def main():
    parser = argparse.ArgumentParser(description="Submit a batch prediction job using Google Cloud AI Platform.")
    parser.add_argument('--project_id', type=str, required=True, help='Google Cloud project ID')
    parser.add_argument('--location', type=str, required=True, help='Location of the job (e.g., us-central1)')
    parser.add_argument('--model_id', type=str, required=True, help='Model ID (e.g., gemini-1.0-pro-002)')
    parser.add_argument('--job_name', type=str, required=True, help='Name of the batch prediction job')
    parser.add_argument('--dataset_name', type=str, required=True, help='BigQuery dataset name')
    parser.add_argument('--input_table_name', type=str, required=True, help='BigQuery input table name')
    parser.add_argument('--output_table_name', type=str, required=True, help='BigQuery output table name')
    parser.add_argument('--input_jsonl_file', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument('--language', type=str, choices=['en', 'sv', 'da', 'nb', 'nn'], default='en', help='Language for the prompt (default: en)')
    parser.add_argument('--dryrun', action='store_true', help='If set, just output the formatted example and exit.')

    args = parser.parse_args()

    chat_prompt = templates[args.language]

    try:
        logging.debug("Preparing input data...")
        prepare_input_data(args.project_id, args.dataset_name, args.input_table_name, args.input_jsonl_file, chat_prompt, args.dryrun)
        if args.dryrun:
            logging.info("Dry run completed. Exiting.")
            return

        logging.debug("Submitting batch prediction job...")
        job_id = submit_batch_prediction_job(args.project_id, args.location, args.model_id, args.job_name, args.dataset_name, args.input_table_name, args.output_table_name)
        logging.info(f"Batch prediction job ID: {job_id}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
