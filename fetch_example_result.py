import argparse
import json
import logging
from google.cloud import bigquery

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fetch an example result from the output table
def fetch_example_result(project_id, dataset_name, output_table_name):
    client = bigquery.Client(project=project_id)
    query = f"SELECT * FROM `{project_id}.{dataset_name}.{output_table_name}` LIMIT 1"
    query_job = client.query(query)
    result = list(query_job.result())[0]
    return dict(result)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Fetch an example result from a BigQuery output table.")
    parser.add_argument('--project_id', type=str, required=True, help='Google Cloud project ID')
    parser.add_argument('--dataset_name', type=str, required=True, help='BigQuery dataset name')
    parser.add_argument('--output_table_name', type=str, required=True, help='BigQuery output table name')

    args = parser.parse_args()

    try:
        logging.debug("Fetching example result...")
        example_result = fetch_example_result(args.project_id, args.dataset_name, args.output_table_name)
        logging.info("Example result:")
        logging.info(json.dumps(example_result, indent=4, default=str))
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
