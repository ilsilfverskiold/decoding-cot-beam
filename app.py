from beam import endpoint, Image, Volume, env
from typing import Dict, Any

# This ensures that these packages are only loaded when the script is running remotely on Beam
if env.is_remote():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cot_decoder import cot_decode
    from question_classifier import get_k_value

# Model parameters
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_PATH = "./cached_models2"

def load_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=torch.float16, cache_dir=CACHE_PATH
    )
    return model, tokenizer

@endpoint(
    secrets=["HF_TOKEN"],
    on_start=load_models,
    name="meta-llama-3-8b-instruct",
    cpu=2,
    memory="32Gi",
    gpu="A100-40",
    image=Image(
        python_version="python3.9",
        python_packages=["torch", "transformers", "accelerate"],
    ),
    volumes=[Volume(name="cached_models2", mount_path=CACHE_PATH)],
)
def generate_text(context: Dict[str, Any], **inputs: Dict[str, Any]) -> Dict[str, Any]:
    # Retrieve model and tokenizer from on_start
    model, tokenizer = context.on_start_value

    # Inputs passed to API
    messages = inputs.pop("messages", None)
    if not messages:
        return {"error": "Please provide messages for text generation."}
    
    # Get adaptive k value based on question complexity
    k = inputs.pop("k", None)  # Allow manual override
    classification_type = None
    if k is None:
        k, classification_type = get_k_value(messages, context)
    
    try:
        output_text, confidence, llm_calls = cot_decode(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            k=k,  # Use adaptive k value
            **inputs  # Pass any additional parameters directly to cot_decode
        )
        
        return {
            "output": output_text,
            "confidence": confidence,
            "complexity_info": {
                "k": k,
                "total_calls": llm_calls + 1,  # Include classification call
                "classification": classification_type
            }
        }
    except Exception as e:
        return {"error": f"Error during generation: {str(e)}"}
