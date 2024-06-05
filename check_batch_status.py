import argparse
import logging
import requests
from google.auth import default
from google.auth.transport.requests import Request

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Authentication
def get_auth_token():
    credentials, _ = default()
    credentials.refresh(Request())
    return credentials.token

# Check the status of the batch prediction job
def check_batch_prediction_job_status(project_id, location, job_id):
    headers = {
        "Authorization": f"Bearer {get_auth_token()}",
        "Content-Type": "application/json"
    }

    response = requests.get(f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/batchPredictionJobs/{job_id}", headers=headers)
    response.raise_for_status()
    job_status = response.json()['state']
    logging.info(f"Batch prediction job status: {job_status}")
    return job_status

# Main function
def main():
    parser = argparse.ArgumentParser(description="Check the status of a batch prediction job.")
    parser.add_argument('--project_id', type=str, required=True, help='Google Cloud project ID')
    parser.add_argument('--location', type=str, required=True, help='Location of the job (e.g., us-central1)')
    parser.add_argument('--job_id', type=str, required=True, help='Batch prediction job ID')

    args = parser.parse_args()

    try:
        logging.debug("Checking batch prediction job status...")
        check_batch_prediction_job_status(args.project_id, args.location, args.job_id)
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
