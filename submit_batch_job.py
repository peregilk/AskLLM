import os
import argparse
import json
import logging
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from google.auth import default
from google.auth.transport.requests import Request
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_auth_token():
    credentials, _ = default()
    credentials.refresh(Request())
    return credentials.token

def insert_rows_to_bigquery(client, dataset_name, table_name, rows, batch_size=500):
    table_ref = client.dataset(dataset_name).table(table_name)
    try:
        client.get_table(table_ref)
        logging.info(f"Table {table_name} already exists.")
    except NotFound:
        schema = [
            bigquery.SchemaField("request", "STRING"),
        ]
        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table)
        logging.info(f"Created table {table_name}")
    
    errors = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        try:
            errors.extend(client.insert_rows_json(table_ref, batch))
        except Exception as e:
            logging.error(f"Error while inserting batch {i // batch_size}: {e}")

    if errors:
        logging.error(f"Encountered errors while inserting rows: {errors}")
    else:
        logging.info(f"Inserted {len(rows)} rows successfully.")

def submit_batch_prediction_job(project_id, location, model_id, job_name, dataset_name, input_table_name, output_table_name):
    ENDPOINT = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/batchPredictionJobs"
    
    auth_token = get_auth_token()
    
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
    response.raise_for_status()
    
    job_id = response.json()['name']
    logging.info(f"Batch prediction job submitted successfully. Job ID: {job_id}")
    return job_id

def process_json_file(json_lines_file, dataset_name, input_table_name, output_table_name, job_name, project_id, location, model_id, language, batch_size):
    client = bigquery.Client()
    
    # Load the templates from the template.json file
    with open('template.json', 'r') as f:
        templates = json.load(f)

    chat_prompt = templates[language]

    with open(json_lines_file, 'r') as file:
        rows = []
        for line in file:
            data = json.loads(line)
            text = data.get('text') or data.get('content')
            if text:
                prompt = chat_prompt.format(content=text)
                row = {"request": json.dumps({
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generation_config": {
                        "temperature": 0.5,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 8192,
                        "response_mime_type": "application/json"
                    },
                    "safety_settings": []
                })}
                rows.append(row)
            else:
                logging.error(f"Field 'text' or 'content' not found in line: {line}")

    logging.info("Inserting rows to BigQuery...")
    insert_rows_to_bigquery(client, dataset_name, input_table_name, rows, batch_size=batch_size)

    logging.info("Submitting batch prediction job...")
    job_id = submit_batch_prediction_job(project_id, location, model_id, job_name, dataset_name, input_table_name, output_table_name)
    return job_id

def main():
    parser = argparse.ArgumentParser(description="Submit a batch prediction job to Vertex AI.")
    parser.add_argument('--project_id', type=str, required=True, help='Google Cloud project ID.')
    parser.add_argument('--dataset_name', type=str, required=True, help='BigQuery dataset name.')
    parser.add_argument('--input_table_name', type=str, required=True, help='BigQuery input table name.')
    parser.add_argument('--output_table_name', type=str, required=True, help='BigQuery output table name.')
    parser.add_argument('--job_name', type=str, required=True, help='Name of the batch prediction job.')
    parser.add_argument('--input_jsonl_file', type=str, required=True, help='Path to the input JSONL file.')
    parser.add_argument('--language', type=str, choices=['en', 'sv', 'da', 'nb', 'nn'], default='en', help='Language for the prompt (default: en).')
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use for the prediction.')
    parser.add_argument('--location', type=str, required=True, help='Location of the Vertex AI endpoint.')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for BigQuery inserts (default: 500).')
    parser.add_argument('--dryrun', action='store_true', help='Output formatted example and do not execute the batch job.')

    args = parser.parse_args()

    if args.dryrun:
        with open(args.input_jsonl_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                text = data.get('text') or data.get('content')
                if text:
                    # Load the templates from the template.json file
                    with open('template.json', 'r') as f:
                        templates = json.load(f)

                    chat_prompt = templates[args.language]
                    prompt = chat_prompt.format(content=text)
                    formatted_example = {
                        "request": json.dumps({
                            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                            "generation_config": {
                                "temperature": 0.5,
                                "top_p": 0.95,
                                "top_k": 40,
                                "max_output_tokens": 8192,
                                "response_mime_type": "application/json"
                            },
                            "safety_settings": []
                        })
                    }
                    logging.info("Dry run - formatted example:")
                    logging.info(json.dumps(formatted_example, indent=4))
                    break
    else:
        process_json_file(args.input_jsonl_file, args.dataset_name, args.input_table_name, args.output_table_name, args.job_name, args.project_id, args.location, args.model_id, args.language, args.batch_size)

if __name__ == "__main__":
    main()
