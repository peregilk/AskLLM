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

Create a BigQuery called Fineweb
```
bq --location=us-central1 mk -d fineweb
```

Submit a batch job to the FineWeb BigQuery. This will return a `job-id`.
```
python submit_batch_job.py --project_id north-390910 --dataset_name fineweb --input_table_name input-table --output_table_name output-table --job_name first-test --input_jsonl_file ../GlotCC/nob-Latn/nob_50.jsonl --language nb --model_id gemini-1.0-pro-002 --location us-central1
```

After a while you can check the status:
```
python check_batch_status.py --project_id north-390910 --location us-central1 --job_id XXXXX
```

And look at the first row in the table:

```
python fetch_example_result.py --project_id north-390910 --dataset_name fineweb --output_table_name output-table
```

