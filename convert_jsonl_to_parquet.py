import os
import json
import argparse
import pandas as pd
import numpy as np

def convert_metadata_to_string(row):
    return json.dumps(row) if isinstance(row, dict) else str(row)

def split_jsonl_to_parquet(input_file, output_dir, num_parts=8):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the JSONLines file into a DataFrame
    df = pd.read_json(input_file, lines=True)
    
    # Shuffle the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Convert metadata column to string
    if 'metadata' in df.columns:
        df['metadata'] = df['metadata'].apply(convert_metadata_to_string)
    
    # Split the DataFrame into parts
    splits = np.array_split(df, num_parts)
    
    # Write each part to a Parquet file
    for i, split_df in enumerate(splits):
        output_file = os.path.join(output_dir, f'train-00000-{i:02x}-00008.parquet')
        split_df.to_parquet(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a JSONLines file into multiple Parquet files.")
    parser.add_argument('--input_file', type=str, required=True, help='Input JSONLines file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the Parquet files.')
    args = parser.parse_args()
    
    split_jsonl_to_parquet(args.input_file, args.output_dir)
