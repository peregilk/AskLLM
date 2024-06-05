import os
import argparse
import requests
from google.cloud import bigquery
from google.auth import default
from google.auth.transport.requests import Request
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Authentication
def get_auth_token():
    credentials, _ = default()
    credentials.refresh(Request())
    return credentials.token

# Check batch prediction job status
def check_batch_prediction_job_status(project_id, location, job_id):
    headers = {
        "Authorization": f"Bearer {get_auth_token()}",
        "Content-Type": "application/json"
    }
    response = requests.get(f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/batchPredictionJobs/{job_id}", headers=headers)
    response.raise_for_status()
    return response.json()

# Retrieve results from BigQuery
def retrieve_results_from_bigquery(project_id, dataset_name, table_name):
    client = bigquery.Client(project=project_id)
    query = f"SELECT * FROM `{project_id}.{dataset_name}.{table_name}` LIMIT 1"
    query_job = client.query(query)
    result = query_job.result()
    return [dict(row) for row in result]

# Main function
def main():
    parser = argparse.ArgumentParser(description="Check batch prediction job status and retrieve results.")
    parser.add_argument('--project_id', type=str, required=True, help='Google Cloud project ID')
    parser.add_argument('--location', type=str, required=True, help='Location of the job (e.g., us-central1)')
    parser.add_argument('--job_id', type=str, required=True, help='Batch prediction job ID')
    parser.add_argument('--dataset_name', type=str, required=True, help='BigQuery dataset name')
    parser.add_argument('--output_table_name', type=str, required=True, help='BigQuery output table name')
    
    args = parser.parse_args()

    try:
        logging.info("Checking batch prediction job status...")
        job_status = check_batch_prediction_job_status(args.project_id, args.location, args.job_id)
        logging.info(f"Batch prediction job status: {job_status['state']}")
        
        if job_status['state'] == 'JOB_STATE_SUCCEEDED':
            logging.info("Retrieving example result from BigQuery...")
            result = retrieve_results_from_bigquery(args.project_id, args.dataset_name, args.output_table_name)
            
            if result:
                logging.info("Example result:")
                example_result = result[0]
                formatted_result = json.dumps(example_result, indent=4, default=str)
                logging.info(formatted_result)
            else:
                logging.info("No results found in the output table.")
        else:
            logging.info("Batch prediction job is not yet complete.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
