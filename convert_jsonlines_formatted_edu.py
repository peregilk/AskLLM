import os
import json
import argparse
import math

def convert_jsonlines_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".jsonl"):
            input_filepath = os.path.join(input_dir, filename)
            output_filepath = os.path.join(output_dir, filename)

            with open(input_filepath, 'r', encoding='utf-8') as infile, open(output_filepath, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    try:
                        record = json.loads(line)
                        educational_score = record.get("educational score", 0)
                        
                        # Check if educational_score is NaN and skip if true
                        if isinstance(educational_score, float) and math.isnan(educational_score):
                            continue
                        
                        new_record = {
                            "text": record.get("content"),
                            "metadata": record,
                            "prompt": filename,
                            "score": int(educational_score)
                        }
                        json.dump(new_record, outfile)
                        outfile.write('\n')
                    except (ValueError, TypeError, KeyError):
                        # Skip lines that result in errors
                        continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONLines files to a new format.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the input JSONLines files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the converted JSONLines files.')
    args = parser.parse_args()
    
    convert_jsonlines_files(args.input_dir, args.output_dir)
