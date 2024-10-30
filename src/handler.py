import os
import runpod
from typing import Any, List, Optional, Sequence, Union, Dict
from pathlib import Path
from transformers import PreTrainedTokenizerBase
from tensorrt_llm import LLM, SamplingParams
from huggingface_hub import login, HfApi
from tensorrt_llm.hlapi import BuildConfig, KvCacheConfig, QuantConfig, QuantAlgo
import logging


# Logging setup
logger = logging.getLogger(__name__)


# Accept huggingface_token
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
        self.model_params = {
            'model': model,
            'enable_build_cache': True,
            'kv_cache_config': KvCacheConfig(),
            'build_config': BuildConfig()
        }
        self.llm = None
    
    def initialize(self):
        """Lazy initialization of the LLM engine"""
        if self.llm is None:
            logger.info("Initializing TensorRT-LLM engine...")
            self.llm = LLM(**self.model_params)
            logger.info("TensorRT-LLM engine initialized successfully")
    
    def save_engine(self, path: str):
        """Save the engine after ensuring it's initialized"""
        self.initialize()  # Ensure engine is initialized before saving
        logger.info("Saving Engine...")
        self.llm.save(path)
        logger.info("...Engine Saved")


def create_engine(model_path: str) -> TRTLLMEngine:
    """Create engine instance without initializing it"""
    return TRTLLMEngine(model=model_path)


async def handler(job: Dict):
    """Handler function that will be used to process jobs."""
    try:
        job_input = job['input']
        
        # Get required parameters from job input
        model_path = job_input.get('model_path')
        if not model_path:
            raise ValueError("model_path is required in job input")
        
        engine_path = job_input.get('path', "/workspace/model")
        push_to_hub = job_input.get('push_to_hf', True)
        hf_repo_id = job_input.get('hf_repo_id')
        
        # Log input parameters
        logger.info(f"Creating engine for model: {model_path}")
        logger.info(f"Engine will be saved to: {engine_path}")
        
        # Create and initialize engine with model from job input
        worker = create_engine(model_path)
        worker.save_engine(engine_path)
        
        if push_to_hub:
            if not hf_repo_id:
                raise ValueError("hf_repo_id is required when push_to_hf is True")
                
            api = HfApi(token=os.environ["HF_TOKEN"])
            api.upload_folder(
                folder_path=engine_path,
                repo_id=hf_repo_id,
                repo_type="model",
            )
            logger.info("TensorRT-LLM Engine pushed to the Hub.")
            return {
                "status": "success", 
                "message": "engine saved and pushed to HF successfully",
                "model_path": model_path,
                "engine_path": engine_path,
                "hf_repo_id": hf_repo_id
            }
        else:
            logger.info(f"Engine saved locally at {engine_path}")
            return {
                "status": "success", 
                "message": "engine saved successfully",
                "model_path": model_path,
                "engine_path": engine_path
            }
            
    except Exception as e:
        logger.error(f"Error during engine processing: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})