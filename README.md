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



Generate NER-dataset:
```
python generic_generate_async.py --jsonl_file /nfsmounts/ficino/lv_ai_2_ficino/perk/NCC2/filtered_above1_5/open_newspapers_no.jsonl --output_jsonl ../AskLLM_datasets/ner_LLM_generated.jsonl --template_file template_ner.txt --max_num_requests 10000
```

