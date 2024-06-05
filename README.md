# AskLLM
Various tools for AskLLM. Just a private repo that happens to be open.


Enable services
```
gcloud services enable bigquery.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com

pip install opentelemetry-api
pip install opentelemetry-instrumentation
````

This method seems to work best:
```
python generate_async.py --jsonl_file ../GlotCC/nob-Latn/nob_90000.jsonl --output_jsonl_file ../GlotCC/nob-Latn/nob_90000_processed.jsonl --language nb
```

