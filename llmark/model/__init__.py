import os

AVAILABLE_MODELS = {
    "llmark_qwen": "LLMArkQwenForCausalLM, LLMArkQwenConfig",
    # "llmark_qwen_moe": "LlavaQwenMoeForCausalLM, LlavaQwenMoeConfig",    
    # Add other models as needed
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except Exception as e:
        print(f"Failed to import {model_name} from llmark.language_model.{model_name}. Error: {e}")
