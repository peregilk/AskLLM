import os
import time
import json
import google.generativeai as genai
import jsonlines
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure the Vertex AI API details
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Load the templates from the template.json file
with open('template.json', 'r') as f:
    templates = json.load(f)

# Define the model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

# Create the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    # safety_settings = Adjust safety settings
    # See https://ai.google.dev/gemini-api/docs/safety-settings
)

def process_json_lines(json_lines_file, language):
    chat_prompt = templates[language]

    with jsonlines.open(json_lines_file, mode='r') as reader:
        for line in reader:
            try:
                text = line.get('text') or line.get('content')
                if text:
                    prompt = chat_prompt.format(content=text)
                    response = model.generate_content(prompt)
                    print(response.text)
                else:
                    logging.error(f"Field 'text' or 'content' not found in line: {line}")
                time.sleep(1)  # Wait to prevent hitting rate limits
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                time.sleep(60)  # Wait for a minute before continuing

def main():
    parser = argparse.ArgumentParser(description="Process a JSONLines file with the Vertex AI API.")
    parser.add_argument('--json_lines_file', type=str, required=True, help='Path to the JSONLines file.')
    parser.add_argument('--language', type=str, choices=['en', 'sv', 'da', 'nb', 'nn'], default='en', help='Language for the prompt (default: en).')

    args = parser.parse_args()

    process_json_lines(args.json_lines_file, args.language)

if __name__ == "__main__":
    main()
