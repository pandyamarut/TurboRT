import os
import runpod
from typing import Any, List, Optional, Sequence, Union, Dict
from pathlib import Path
from transformers import PreTrainedTokenizerBase
from tensorrt_llm import LLM, SamplingParams
from huggingface_hub import login
from tensorrt_llm.hlapi import BuildConfig, KvCacheConfig, QuantConfig, QuantAlgo
import logging
from huggingface_hub import HfApi


#logging
logger = logging.getLogger(__name__)


#Accept huggingface_token
hf_token = os.environ["HF_TOKEN"]
login(token=hf_token)



class TRTLLMEngine:
    def __init__(self,
                    model: str, 
                    tokenizer: Optional[Union[str, Path,
                                           PreTrainedTokenizerBase]] = None,
                    skip_tokenizer_init: bool = False,
                    tensor_parallel_size: int = 1,
                    dtype: str = "auto", 
                    trust_remote_code: bool = False,
                    revision: Optional[str] = None,
                    tokenizer_revision: Optional[str] = None,
                    **kwargs: Any
                ):
        self.llm = LLM(model=model_path, enable_build_cache=True, kv_cache_config=KvCacheConfig(), build_config=BuildConfig())
    
    def save_engine(self, path: str):
        logger.info("Save Engine...")
        self.llm.save(path)
        logger.info("...Engine Saved")
    

 
# Initialize the worker outside the handler
# This ensures the model is loaded only once when the serverless function starts
# this path is hf model "<org_name>/model_name" egs: meta-llama/Meta-Llama-3.1-8B-Instruct
model_path = os.environ["MODEL_PATH"]
tokenizer = os.environ["TOKENIZER"]
skip_tokenizer_init = os.environ["SKIP_TOKENIZER_INIT"]
tensor_parallel_size = os.environ["TENSOR_PARALLEL_SIZE"]
dtype = os.environ["DTYPE"]
trust_remote_code = os.environ["TRUST_REMOTE_CODE"]
revision = os.environ["REVISION"]
tokenizer_revision = os.environ["TOKENIZER_REVISION"]

worker = TRTLLMEngine(model_path, tokenizer, skip_tokenizer_init, tensor_parallel_size, dtype, trust_remote_code, revision, tokenizer_revision)


async def handler(job: Dict):
    """Handler function that will be used to process jobs."""
    job_input = job['input']
    path = job_input.get('path', "/workspace/model")
    push_to_hub = job.get('push_to_hf', True)
    hf_repo_id = job.get("hf_repo_id")
    
    try:
        worker.save_engine(path)
        if push_to_hub == True:
            api = HfApi(token=os.environ["HF_TOKEN"])
            api.upload_folder(
                folder_path=path,
                repo_id=hf_repo_id,
                repo_type="model",
            )
            logger.info("Tensorrt-LLM Engine pushed to the Hub.")
            return {"status": "success", "message": "engine saved and pushed to HF successfully"}
        else:
            logger.info(f"Engine saved locally at {path}")
            return {"status": "success", "message": "engine saved successfully"}
    except: 
        return {"status": "error", "message": "errored"}
            
    

runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})