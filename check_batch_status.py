import argparse
import logging
import json
import google.auth
import google.auth.transport.requests
import requests
from google.cloud import bigquery
from google.cloud import logging as cloud_logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_batch_prediction_job_status(project_id, location, job_id):
    ENDPOINT = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/batchPredictionJobs/{job_id}"
    
    credentials, _ = google.auth.default()
    credentials.refresh(google.auth.transport.requests.Request())
    auth_token = credentials.token

    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }

    response = requests.get(ENDPOINT, headers=headers)
    response.raise_for_status()
    
    job_status = response.json()
    state = job_status.get('state', 'Unknown')
    logging.info(f"Batch prediction job status: {state}")

    if state == 'JOB_STATE_FAILED':
        error = job_status.get('error', {})
        logging.error(f"Job failed with error: {error.get('message', 'No error message available')}")
        logging.error(f"Error details: {json.dumps(error, indent=4)}")
        return None, job_status

    return state, job_status

def fetch_logs(project_id, job_id):
    logging_client = cloud_logging.Client(project=project_id)
    logger = logging_client.logger('batch_prediction_job')
    
    # Filter logs by job_id
    filter_str = f'resource.type="ml_job" AND resource.labels.job_id="{job_id}"'
    entries = logging_client.list_entries(filter_=filter_str)
    
    for entry in entries:
        logging.info(entry.payload)

def query_bigquery(dataset_name, output_table_name):
    client = bigquery.Client()
    query = f"SELECT * FROM `{dataset_name}.{output_table_name}` LIMIT 1"
    query_job = client.query(query)

    results = query_job.result()
    for row in results:
        logging.info(json.dumps(dict(row), indent=4, default=str))

def main():
    parser = argparse.ArgumentParser(description="Check the status of a batch prediction job on Vertex AI and query results from BigQuery.")
    parser.add_argument('--project_id', type=str, required=True, help='Google Cloud project ID.')
    parser.add_argument('--location', type=str, required=True, help='Location of the Vertex AI endpoint.')
    parser.add_argument('--job_id', type=str, help='ID of the batch prediction job.')
    parser.add_argument('--dataset_name', type=str, help='BigQuery dataset name.')
    parser.add_argument('--output_table_name', type=str, help='BigQuery output table name.')

    args = parser.parse_args()

    if args.job_id:
        state, job_status = check_batch_prediction_job_status(args.project_id, args.location, args.job_id)
        if state == 'JOB_STATE_FAILED':
            fetch_logs(args.project_id, args.job_id)
        elif state == 'JOB_STATE_SUCCEEDED' and args.dataset_name and args.output_table_name:
            query_bigquery(args.dataset_name, args.output_table_name)
    elif args.dataset_name and args.output_table_name:
        query_bigquery(args.dataset_name, args.output_table_name)
    else:
        logging.error("You must provide either job_id to check status or dataset_name and output_table_name to query results.")

if __name__ == "__main__":
    main()
