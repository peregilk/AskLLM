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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure the Vertex AI API details
PROJECT_ID = "north-390910"
LOCATION = "us-central1"
MODEL_ID = "gemini-1.5-flash-001"
ENDPOINT = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:generateContent"

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

# Fetch and cache the authentication token once
auth_token = None

def get_auth_token():
    global auth_token
    if auth_token is None:
        credentials, _ = default()
        credentials.refresh(Request())
        auth_token = credentials.token
    return auth_token

def send_request(prompt):
    headers = {
        "Authorization": f"Bearer {get_auth_token()}",
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }],
        "generation_config": generation_config,
        "safety_settings": safety_settings
    }
    response = requests.post(ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

def process_json_lines(json_lines_file, output_file, num_examples, max_requests_per_minute, language, text_field, verbose, wait_time):
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    chat_prompt = templates[language]

    try:
        # Count the number of lines already processed in the output file
        if os.path.exists(output_file):
            with jsonlines.open(output_file, mode='r') as reader:
                processed_lines_count = sum(1 for _ in reader)
        else:
            processed_lines_count = 0

        logging.info(f"Starting from line: {processed_lines_count}")

        with jsonlines.open(json_lines_file, mode='r') as reader:
            lines = list(reader)

        total_lines = min(processed_lines_count + num_examples, len(lines))
        
        with jsonlines.open(output_file, mode='a') as writer:
            with tqdm(total=total_lines - processed_lines_count, desc="Processing lines") as pbar:
                for idx in range(processed_lines_count, total_lines):
                    line = lines[idx]

                    if "educational score" in line:
                        logging.debug(f"Line already processed, skipping: {line}")
                        writer.write(line)
                        pbar.update(1)
                        continue

                    if text_field not in line:
                        logging.error(f"Field '{text_field}' not found in line {idx+1}. Make sure the input JSONL file contains this field.")
                        writer.write(line)
                        pbar.update(1)
                        continue

                    retries = 0
                    while retries < 5:
                        try:
                            input_text = line[text_field]
                            prompt = chat_prompt.format(content=input_text)
                            response = send_request(prompt)
                            response_json_str = response['candidates'][0]['content']['parts'][0]['text']
                            response_json = json.loads(response_json_str)
                            logging.debug(f"Response JSON: {response_json}")
                            line["justification"] = response_json.get("reason", "No justification found")
                            line["educational score"] = response_json.get("educational score", 0)
                            logging.debug(f"Writing line: {line}")
                            writer.write(line)
                            logging.debug(f"Written line: {line}")
                            pbar.update(1)
                            time.sleep(wait_time)  # Wait for the specified time between each request
                            break
                        except Exception as e:
                            retries += 1
                            logging.error(f"An error occurred while processing line {idx+1} (attempt {retries}): {e}")
                            if retries >= 5:
                                logging.error("Maximum retry limit reached. Exiting the script.")
                                exit(1)
                    
                    if (idx + 1) % max_requests_per_minute == 0:
                        logging.info(f"Processed {idx + 1} entries. Waiting for a minute to respect rate limit.")
                        time.sleep(60)  # Wait for a minute to respect rate limit

    except Exception as e:
        logging.error(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process a JSONLines file with the Vertex AI API.")
    parser.add_argument('--json_lines_file', type=str, required=True, help='Path to the JSONLines file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSONLines file.')
    parser.add_argument('--num_examples', type=int, default=100, help='Number of requests to process (default: 100).')
    parser.add_argument('--max_requests_per_minute', type=int, default=1000, help='Maximum number of requests per minute (default: 1000).')
    parser.add_argument('--language', type=str, choices=['en', 'sv', 'da', 'nb', 'nn'], default='en', help='Language for the prompt (default: en).')
    parser.add_argument('--text_field', type=str, default='text', help='Field in JSON lines containing the text (default: text).')
    parser.add_argument('--wait_time', type=float, default=0, help='Time to wait between requests in seconds (default: 0).')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging.')

    args = parser.parse_args()

    process_json_lines(args.json_lines_file, args.output_file, args.num_examples, args.max_requests_per_minute, args.language, args.text_field, args.verbose, args.wait_time)

if __name__ == "__main__":
    main()
