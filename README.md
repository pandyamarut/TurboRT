# TensorRT-LLM Engine Builder

A serverless RunPod worker for building and saving TensorRT-LLM engines with automatic pushing to HuggingFace Hub.

## Overview

This worker builds TensorRT-LLM engines from HuggingFace models and optionally pushes them to the HuggingFace Hub. It's designed to run as a serverless endpoint on RunPod, making it easy to generate optimized TensorRT engines for your models.

## Usage

### Prerequisites

- RunPod account with API access
- HuggingFace account and API token

### Making a Request

Send a POST request to your RunPod endpoint with the following JSON structure:

```json
{
  "input": {
    "model_path": "meta-llama/Llama-2-7b",    // HuggingFace model to convert
    "hf_repo_id": "your-username/your-repo",  // Where to push the built engine
    "push_to_hf": "True",                     // Whether to push to HuggingFace
    "path": "/workspace/model"                // Optional: Local save path
  }
}
```

#### Parameters

- `model_path` (required): HuggingFace model path to convert
- `hf_repo_id`: Your HuggingFace repository ID where the engine will be pushed
- `push_to_hf`: Set to "True" to push the engine to HuggingFace Hub
- `path`: Optional local path to save the engine (defaults to "/workspace/model")

#### Example using cURL

```bash
curl -X POST "https://api.runpod.ai/v2/{your-endpoint-id}/run" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
     -d '{
           "input": {
             "model_path": "meta-llama/Llama-2-7b",
             "hf_repo_id": "pandyamarut/test_trt_llm",
             "push_to_hf": "True"
           }
         }'
```

#### Example using Python

```python
import requests

RUNPOD_API_KEY = "your-api-key"
ENDPOINT_ID = "your-endpoint-id"

url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"

payload = {
    "input": {
        "model_path": "meta-llama/Llama-2-7b",
        "hf_repo_id": "pandyamarut/test_trt_llm",
        "push_to_hf": "True"
    }
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {RUNPOD_API_KEY}"
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

### Response Format

Success Response:
```json
{
    "status": "success",
    "message": "engine saved and pushed to HF successfully",
    "model_path": "meta-llama/Llama-2-7b",
    "engine_path": "/workspace/model",
    "hf_repo_id": "pandyamarut/test_trt_llm"
}
```

Error Response:
```json
{
    "status": "error",
    "message": "error message details"
}
```

## Environment Variables

The worker container requires the following environment variables:

```bash
HF_TOKEN          # Your HuggingFace API token
```

## Container Deployment

The worker is designed to be deployed as a serverless endpoint on RunPod. Make sure your RunPod environment has:

1. Appropriate GPU resources for your model
2. TensorRT-LLM dependencies installed
3. HF_TOKEN environment variable configured

## Known Limitations

- Engine building requires significant GPU memory
- Process time varies based on model size
- Only supports models compatible with TensorRT-LLM

## Example Repositories

- Built Engine Example: [pandyamarut/test_trt_llm](https://huggingface.co/pandyamarut/test_trt_llm)

## License

This project is licensed under the MIT License.