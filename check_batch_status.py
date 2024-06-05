import argparse
import logging
import json
import google.auth
import google.auth.transport.requests
import requests

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
        
        # Fetching more detailed error information
        details = error.get('details', [])
        if details:
            for detail in details:
                logging.error(f"Error detail: {json.dumps(detail, indent=4)}")
        return state, job_status

    return state, job_status

def main():
    parser = argparse.ArgumentParser(description="Check the status of a batch prediction job.")
    parser.add_argument('--project_id', type=str, required=True, help='Google Cloud project ID.')
    parser.add_argument('--location', type=str, required=True, help='Location of the Vertex AI endpoint.')
    parser.add_argument('--job_id', type=str, required=True, help='ID of the batch prediction job.')

    args = parser.parse_args()

    state, job_status = check_batch_prediction_job_status(args.project_id, args.location, args.job_id)
    
    if state == 'JOB_STATE_SUCCEEDED':
        logging.info("Batch prediction job completed successfully.")
    elif state == 'JOB_STATE_FAILED':
        logging.error("Batch prediction job failed.")
    else:
        logging.info("Batch prediction job is still in progress or in an unknown state.")

if __name__ == "__main__":
    main()
