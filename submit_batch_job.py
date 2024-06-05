import os
import json
import argparse
import requests
from google.cloud import bigquery
from google.auth import default
from google.auth.transport.requests import Request
import logging
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Authentication
def get_auth_token():
    credentials, _ = default()
    credentials.refresh(Request())
    return credentials.token

# Prepare the BigQuery table for input data
def prepare_input_data(input_jsonl_file, project_id, dataset_name, input_table_name):
    logging.debug("Preparing input data...")
    bq_client = bigquery.Client(project=project_id)
    dataset_ref = bq_client.dataset(dataset_name)
    table_ref = dataset_ref.table(input_table_name)
    
    # Define table schema
    schema = [
        bigquery.SchemaField("request", "STRING", mode="REQUIRED")
    ]
    
    # Create table if it doesn't exist
    try:
        table = bq_client.get_table(table_ref)
        logging.info(f"Table {input_table_name} already exists.")
    except Exception:
        table = bigquery.Table(table_ref, schema=schema)
        table = bq_client.create_table(table)
        logging.info(f"Created table {table.table_id}")

    # Load input data from JSONL file
    rows_to_insert = []
    with open(input_jsonl_file, 'r') as file:
        for line in file:
            try:
                rows_to_insert.append({"request": json.dumps({"contents": [{"role": "user", "parts": [{"text": json.loads(line)['content']}] }] })})
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing line: {line}. Error: {e}")

    if rows_to_insert:
        # Insert rows in batches
        BATCH_SIZE = 500
        for i in range(0, len(rows_to_insert), BATCH_SIZE):
            batch = rows_to_insert[i:i + BATCH_SIZE]
            errors = None
            retries = 0
            while retries < 5:
                try:
                    errors = bq_client.insert_rows_json(table, batch)
                    if errors == []:
                        logging.info(f"Batch {i // BATCH_SIZE} inserted successfully.")
                        break
                    else:
                        logging.error(f"Encountered errors while inserting batch {i // BATCH_SIZE}: {errors}")
                except Exception as e:
                    logging.error(f"Exception during batch insertion: {e}")
                retries += 1
                time.sleep(2 ** retries)  # Exponential backoff
            if retries == 5:
                logging.error(f"Failed to insert batch {i // BATCH_SIZE} after 5 retries.")
    else:
        logging.error("No valid rows to insert.")

# Submit batch prediction job
def submit_batch_prediction_job(project_id, location, model_id, job_name, dataset_name, input_table_name, output_table_name):
    logging.debug("Submitting batch prediction job...")
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
    logging.debug(f"Payload: {json.dumps(payload, indent=2)}")
    response = requests.post(f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/batchPredictionJobs", headers=headers, json=payload)
    
    if response.status_code != 200:
        logging.error(f"HTTPError: {response.status_code} {response.reason}")
        logging.error(f"Response content: {response.content.decode('utf-8')}")
    response.raise_for_status()
    return response.json()["name"]

def main():
    parser = argparse.ArgumentParser(description="Submit a batch prediction job with Vertex AI and BigQuery.")
    parser.add_argument('--project_id', type=str, required=True, help='Google Cloud project ID.')
    parser.add_argument('--location', type=str, default='us-central1', help='Google Cloud location (default: us-central1).')
    parser.add_argument('--model_id', type=str, default='gemini-1.5-pro', help='Vertex AI model ID (default: gemini-1.5-pro-001).')
    parser.add_argument('--dataset_name', type=str, required=True, help='BigQuery dataset name.')
    parser.add_argument('--input_table_name', type=str, required=True, help='BigQuery input table name.')
    parser.add_argument('--output_table_name', type=str, required=True, help='BigQuery output table name.')
    parser.add_argument('--job_name', type=str, required=True, help='Batch prediction job name.')
    parser.add_argument('--input_jsonl_file', type=str, required=True, help='Path to the input JSONL file.')
    
    args = parser.parse_args()

    prepare_input_data(args.input_jsonl_file, args.project_id, args.dataset_name, args.input_table_name)
    job_id = submit_batch_prediction_job(args.project_id, args.location, args.model_id, args.job_name, args.dataset_name, args.input_table_name, args.output_table_name)
    logging.info(f"Batch prediction job submitted successfully. Job ID: {job_id}")

if __name__ == "__main__":
    main()
