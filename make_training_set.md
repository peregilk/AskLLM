# Make training set
'''
#!/bin/bash

# Create the target directory if it doesn't exist
mkdir -p ../linguistic_jsonl

# Extract the last 2500 lines from each file and save them
tail -n 2500 da_processed.jsonl > ../linguistic_jsonl/validation_da.jsonl
tail -n 2500 en_processed.jsonl > ../linguistic_jsonl/validation_en.jsonl
tail -n 2500 no_processed.jsonl > ../linguistic_jsonl/validation_no.jsonl
tail -n 2500 sv_processed.jsonl > ../linguistic_jsonl/validation_sv.jsonl

# Combine the validation files and shuffle them
cat ../linguistic_jsonl/validation_da.jsonl ../linguistic_jsonl/validation_en.jsonl ../linguistic_jsonl/validation_no.jsonl ../linguistic_jsonl/validation_sv.jsonl | shuf > ../linguistic_jsonl/validation.jsonl

# Remove the last 2500 lines from each file and save the remaining lines to temp files
head -n -2500 da_processed.jsonl > temp_da.jsonl
head -n -2500 en_processed.jsonl > temp_en.jsonl
head -n -2500 no_processed.jsonl > temp_no.jsonl
head -n -2500 sv_processed.jsonl > temp_sv.jsonl

# Combine the remaining lines and shuffle them
cat temp_da.jsonl temp_en.jsonl temp_no.jsonl temp_sv.jsonl | shuf > ../linguistic_jsonl/train.jsonl

# Clean up the temporary files
rm temp_da.jsonl temp_en.jsonl temp_no.jsonl temp_sv.jsonl
'''
