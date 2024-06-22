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

def trim_text(text, max_length):
    trimmed_text = text[:max_length]
    last_period = trimmed_text.rfind('.')
    if last_period != -1:
        return trimmed_text[:last_period + 1]
    return trimmed_text

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

async def process_batch(session, batch, endpoint, template, total_words, max_length):
    tasks = []

    for idx, line in batch.iterrows():
        input_text = trim_text((line.get('text') or line.get('content')), max_length)
        prompt = template.replace("{content}", input_text)
        logging.info(f"Formatted prompt for line {idx}.")
        total_words += len(prompt.split())
        tasks.append(send_request(session, endpoint, prompt, idx))
    
    responses = await asyncio.gather(*tasks)
    
    for idx, response in responses:
        if response == 'rate_limit':
            logging.warning(f"Rate limit hit for request {idx}. Retrying...")
            return False, total_words, None  # Indicate that rate limit was hit and retry is needed
        if response == 'unauthorized':
            raise Exception("Unauthorized request. Check your credentials.")
        if response is not None:
            try:
                logging.debug(f"Response for request {idx}: {response}")
                response_text = response['candidates'][0]['content']['parts'][0]['text']
                batch.loc[idx, 'askLLMresult'] = response_text
                total_words += len(response_text.split())
            except Exception as e:
                logging.error(f"Failed to process response for request {idx} due to unexpected error: {e}")
                logging.error(f"Response structure for request {idx}: {json.dumps(response, indent=2)}")
        else:
            logging.error(f"No response for request {idx}")
    return True, total_words, batch

async def process_json_lines(jsonl_file, output_jsonl_file, template_file, dryrun, model_id, wait_rate_limit, wait_time, max_length, max_num_requests):
    total_words = 0
    retry_limit = 5
    retry_attempts = 0
    request_count = 0

    logging.info(f"Loading lines from {jsonl_file}...")
    df = pd.read_json(jsonl_file, lines=True)
    logging.info(f"Loaded {len(df)} lines from the file.")

    endpoint = ENDPOINT_TEMPLATE.format(location=LOCATION, project_id=PROJECT_ID, model_id=model_id)

    with open(template_file, 'r') as f:
        template = f.read()
        if '{content}' not in template:
            raise ValueError("Template file must contain '{content}' placeholder.")
        
        # Replace curly quotes with straight quotes
        template = template.replace('“', '"').replace('”', '"')
        logging.info(f"Template loaded successfully.")

    if dryrun:
        first_line = df.iloc[0]
        input_text = trim_text((first_line.get('text') or first_line.get('content')), max_length)
        chat_prompt = template.replace("{content}", input_text)
        print(f"Dryrun: {chat_prompt}")
        return

    total_requests = min(len(df), max_num_requests) if max_num_requests else len(df)
    
    with jsonlines.open(output_jsonl_file, mode='w') as writer:
        with tqdm(total=total_requests, desc="Processing lines") as pbar:
            for start in range(0, len(df), BATCH_SIZE):
                if max_num_requests and request_count >= max_num_requests:
                    logging.info(f"Reached the maximum number of requests: {max_num_requests}. Stopping.")
                    break

                end = min(start + BATCH_SIZE, len(df))
                batch_size = end - start
                if max_num_requests:
                    batch_size = min(batch_size, max_num_requests - request_count)

                batch = df[start:start + batch_size].copy()  # Make a copy to avoid SettingWithCopyWarning
                get_auth_token()  # Refresh the token before processing each batch
                async with aiohttp.ClientSession() as session:
                    success, total_words, processed_batch = await process_batch(session, batch, endpoint, template, total_words, max_length)

                if not success:
                    if retry_attempts < retry_limit:
                        retry_attempts += 1
                        logging.warning(f"Retrying batch {start}-{end} due to rate limit.")
                        await asyncio.sleep(wait_rate_limit)  # Wait for wait_rate_limit seconds before retrying
                        continue
                    else:
                        logging.error(f"Batch {start}-{end} failed after {retry_limit} attempts.")
                        break
                
                retry_attempts = 0  # Reset retry attempts after successful batch
                writer.write_all(processed_batch.to_dict(orient='records'))
                pbar.update(len(batch))
                logging.info(f"{len(batch)} of {batch_size} succeeded. Waiting {wait_time} seconds before next batch.")
                await asyncio.sleep(wait_time)

                request_count += len(batch)

                if max_num_requests and request_count >= max_num_requests:
                    logging.info(f"Reached the maximum number of requests: {max_num_requests}. Stopping.")
                    break

            # Process and write the last batch if it's not a multiple of BATCH_SIZE
            if end < len(df) and (not max_num_requests or request_count < max_num_requests):
                last_batch_size = (max_num_requests - request_count) if max_num_requests else len(df) - end
                last_batch = df[end:end + last_batch_size].copy()
                async with aiohttp.ClientSession() as session:
                    success, total_words, processed_last_batch = await process_batch(session, last_batch, endpoint, template, total_words, max_length)
                if success:
                    writer.write_all(processed_last_batch.to_dict(orient='records'))
                    pbar.update(len(last_batch))
                    logging.info(f"Processed the last batch of {len(last_batch)} lines.")

    logging.info(f"Total words processed (input + output): {total_words}")


def main():
    parser = argparse.ArgumentParser(description="Process a JSONLines file with the Vertex AI API.")
    parser.add_argument('--jsonl_file', type=str, required=True, help='Path to the JSONLines file.')
    parser.add_argument('--output_jsonl_file', type=str, required=True, help='Path to the output JSONLines file.')
    parser.add_argument('--template_file', type=str, required=True, help='Path to the template file.')
    parser.add_argument('--dryrun', action='store_true', help='Perform a dry run without sending requests.')
    parser.add_argument('--model_id', type=str, default='gemini-1.5-flash-001', help='Model ID to use for the API.')
    parser.add_argument('--wait_rate_limit', type=int, default=300, help='Rate limit in seconds for retrying after hitting rate limit (default: 300 seconds).')
    parser.add_argument('--wait_time', type=int, default=65, help='Wait time in seconds between batches (default: 65 seconds).')
    parser.add_argument('--max_length', type=int, default=1000, help='Maximum length of input text to be processed (default: 1000 characters).')
    parser.add_argument('--max_num_requests', type=int, help='Maximum number of requests to process.')

    args = parser.parse_args()

    asyncio.run(process_json_lines(args.jsonl_file, args.output_jsonl_file, args.template_file, args.dryrun, args.model_id, args.wait_rate_limit, args.wait_time, args.max_length, args.max_num_requests))

if __name__ == "__main__":
    main()
