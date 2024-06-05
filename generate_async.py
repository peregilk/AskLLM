import os
import json
import aiohttp
import asyncio
import logging
import argparse
import pandas as pd
import jsonlines
from google.auth.transport.requests import Request
from google.auth import default
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Vertex AI API details
PROJECT_ID = "north-390910"
LOCATION = "us-central1"
ENDPOINT_TEMPLATE = "https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:generateContent"

# Load the templates
with open('template.json', 'r') as f:
    TEMPLATES = json.load(f)

# Model configuration
GENERATION_CONFIG = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

# Batch size and wait time
BATCH_SIZE = 200
WAIT_TIME = 65  # Wait time between batches in seconds

# Fetch and cache the authentication token
auth_token = None
credentials = None

def get_auth_token():
    global auth_token, credentials
    if credentials is None:
        credentials, _ = default()
    credentials.refresh(Request())
    auth_token = credentials.token
    logging.info("Authentication token refreshed.")
    return auth_token

async def send_request(session, endpoint, prompt, idx):
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }],
        "generation_config": GENERATION_CONFIG,
    }
    try:
        async with session.post(endpoint, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return idx, result
            elif response.status == 401:
                logging.error(f"Request {idx} failed with status: 401 Unauthorized. Check your credentials.")
                return idx, 'unauthorized'
            elif response.status == 429:
                logging.warning(f"Request {idx} hit rate limit. Waiting for 5 minutes.")
                await asyncio.sleep(300)  # Wait for 5 minutes before retrying
                return idx, 'rate_limit'
            else:
                logging.error(f"Request {idx} failed with status: {response.status}")
                return idx, None
    except Exception as e:
        logging.error(f"Request {idx} failed with exception: {e}")
        return idx, None

async def process_batch(session, batch, endpoint, language, total_words):
    tasks = []
    chat_prompt = TEMPLATES[language]

    for idx, line in batch.iterrows():
        input_text = line.get('text') or line.get('content')
        prompt = chat_prompt.format(content=input_text)
        total_words += len(prompt.split())
        tasks.append(send_request(session, endpoint, prompt, idx))
    
    responses = await asyncio.gather(*tasks)
    
    for idx, response in responses:
        if response == 'rate_limit':
            return False, total_words, None  # Indicate that rate limit was hit and retry is needed
        if response == 'unauthorized':
            raise Exception("Unauthorized request. Check your credentials.")
        if response is not None:
            try:
                response_json_str = response['candidates'][0]['content']['parts'][0]['text']
                response_json = json.loads(response_json_str)
                batch.loc[idx, 'reason'] = response_json.get('reason', 'No reason found')
                batch.loc[idx, 'educational score'] = response_json.get('educational score', 0)
                total_words += len(response_json_str.split())
            except Exception as e:
                logging.error(f"Failed to process response for request {idx}: {e}")

    return True, total_words, batch

async def process_json_lines(jsonl_file, output_jsonl_file, language, dryrun, model_id):
    total_words = 0
    retry_limit = 5
    retry_attempts = 0

    logging.info(f"Loading lines from {jsonl_file}...")
    df = pd.read_json(jsonl_file, lines=True)
    logging.info(f"Loaded {len(df)} lines from the file.")

    endpoint = ENDPOINT_TEMPLATE.format(location=LOCATION, project_id=PROJECT_ID, model_id=model_id)

    if dryrun:
        first_line = df.iloc[0]
        input_text = first_line.get('text') or first_line.get('content')
        chat_prompt = TEMPLATES[language].format(content=input_text)
        print(f"Dryrun: {chat_prompt}")
        return

    with jsonlines.open(output_jsonl_file, mode='w') as writer:
        with tqdm(total=len(df), desc="Processing lines") as pbar:
            for start in range(0, len(df), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(df))
                batch = df[start:end].copy()  # Make a copy to avoid SettingWithCopyWarning
                get_auth_token()  # Refresh the token before processing each batch
                async with aiohttp.ClientSession() as session:
                    success, total_words, processed_batch = await process_batch(session, batch, endpoint, language, total_words)

                if not success:
                    if retry_attempts < retry_limit:
                        retry_attempts += 1
                        logging.warning(f"Retrying batch {start}-{end} due to rate limit.")
                        await asyncio.sleep(300)  # Wait for 5 minutes before retrying
                        continue
                    else:
                        logging.error(f"Batch {start}-{end} failed after {retry_limit} attempts.")
                        break
                
                retry_attempts = 0  # Reset retry attempts after successful batch
                writer.write_all(processed_batch.to_dict(orient='records'))
                pbar.update(len(batch))
                logging.info(f"{len(batch)} of {BATCH_SIZE} succeeded. Waiting {WAIT_TIME} seconds before next batch.")
                await asyncio.sleep(WAIT_TIME)

    logging.info(f"Total words processed (input + output): {total_words}")

def main():
    parser = argparse.ArgumentParser(description="Process a JSONLines file with the Vertex AI API.")
    parser.add_argument('--jsonl_file', type=str, required=True, help='Path to the JSONLines file.')
    parser.add_argument('--output_jsonl_file', type=str, required=True, help='Path to the output JSONLines file.')
    parser.add_argument('--language', type=str, choices=['en', 'sv', 'da', 'nb', 'nn'], default='en', help='Language for the prompt (default: en).')
    parser.add_argument('--dryrun', action='store_true', help='Perform a dry run without sending requests.')
    parser.add_argument('--model_id', type=str, default='gemini-1.5-flash-001', help='Model ID to use for the API.')

    args = parser.parse_args()

    asyncio.run(process_json_lines(args.jsonl_file, args.output_jsonl_file, args.language, args.dryrun, args.model_id))

if __name__ == "__main__":
    main()
