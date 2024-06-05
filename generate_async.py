import os
import json
import aiohttp
import asyncio
import logging
import argparse
import pandas as pd
import time
from google.auth.transport.requests import Request
from google.auth import default
from tqdm import tqdm

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
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,  # Corrected top_k value
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

auth_token = None

def get_auth_token():
    global auth_token
    if auth_token is None:
        credentials, _ = default()
        credentials.refresh(Request())
        auth_token = credentials.token
    return auth_token

async def send_request(session, prompt, idx):
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }],
        "generation_config": generation_config,
        "safety_settings": []
    }
    headers = {
        "Authorization": f"Bearer {get_auth_token()}",
        "Content-Type": "application/json",
    }
    async with session.post(ENDPOINT, headers=headers, json=payload) as response:
        if response.status == 200:
            result = await response.json()
            response_json_str = result['candidates'][0]['content']['parts'][0]['text']
            try:
                response_json = json.loads(response_json_str)
                return response_json
            except json.JSONDecodeError as e:
                logging.error(f"JSONDecodeError for request {idx}: {e}")
                logging.error(f"Response content: {response_json_str}")
                return None
        else:
            logging.error(f"Request {idx} failed with status: {response.status}")
            if response.status == 401:  # Token expired
                auth_token = None  # Force refresh the token
            return None

async def process_batch(batch, language, total_words):
    chat_prompt = templates[language]
    async with aiohttp.ClientSession() as session:
        tasks = []
        for idx, row in enumerate(batch):
            input_text = row.get('text', row.get('content', None))
            if input_text:
                prompt = chat_prompt.format(content=input_text)
                tasks.append(asyncio.ensure_future(send_request(session, prompt, idx)))
                total_words += len(input_text.split())
        results = await asyncio.gather(*tasks)
    for result in results:
        if result:
            total_words += len(result['reason'].split())
    return results, total_words

async def process_json_lines(jsonl_file, output_jsonl_file, language):
    df = pd.read_json(jsonl_file, lines=True)
    logging.info(f"Loaded {len(df)} lines from the file.")
    batch_size = 100
    total_words = 0

    with tqdm(total=len(df), desc="Processing lines") as pbar:
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start:start + batch_size].to_dict(orient='records')
            start_time = time.time()
            retries = 0

            while retries < 5:
                results, total_words = await process_batch(batch, language, total_words)
                if all(result is not None for result in results):
                    # All requests were successful, write to output file
                    for i, result in enumerate(results):
                        if result:
                            batch[i]['reason'] = result.get('reason')
                            batch[i]['educational score'] = result.get('educational score', 0)
                    with open(output_jsonl_file, 'a') as f:
                        for item in batch:
                            f.write(json.dumps(item) + '\n')
                    break
                else:
                    retries += 1
                    logging.error(f"Batch {start // batch_size + 1} failed. Retrying {retries}/5...")
                    await asyncio.sleep(60)  # Wait for a minute before retrying

            if retries >= 5:
                logging.error(f"Batch {start // batch_size + 1} failed after 5 retries. Skipping batch.")

            elapsed_time = time.time() - start_time
            if elapsed_time < 0.2:
                await asyncio.sleep(0.2 - elapsed_time)
            else:
                logging.info(f"Batch {start // batch_size + 1} processed in {elapsed_time} seconds")

            pbar.update(batch_size)
            if start + batch_size < len(df):
                logging.info("Waiting for 0.2 seconds before the next batch...")
                await asyncio.sleep(0.2)

    logging.info(f"Total words processed: {total_words}")

def main():
    parser = argparse.ArgumentParser(description="Process a JSONLines file with the Vertex AI API.")
    parser.add_argument('--jsonl_file', type=str, required=True, help='Path to the JSONLines file.')
    parser.add_argument('--output_jsonl_file', type=str, required=True, help='Path to the output JSONLines file.')
    parser.add_argument('--language', type=str, choices=['en', 'sv', 'da', 'nb', 'nn'], default='en', help='Language for the prompt (default: en).')

    args = parser.parse_args()

    asyncio.run(process_json_lines(args.jsonl_file, args.output_jsonl_file, args.language))

if __name__ == "__main__":
    main()
