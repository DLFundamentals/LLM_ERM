import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, LlamaForCausalLM, BloomModel, BloomConfig
from typing import Dict, Any

# =============================================================================
# Abstract Base Classes for Reusability
# =============================================================================

class BaseFinetuningModel(nn.Module):
    """
    An abstract base class for fine-tuning pretrained causal language models.
    
    This class handles the common logic for:
    1. Loading a pretrained model from Hugging Face.
    2. Enabling gradient checkpointing for memory efficiency.
    3. Freezing all model parameters.
    4. Unfreezing the top N layers for fine-tuning.
    5. Replacing the language modeling head with a linear layer for binary classification.
    """
    def __init__(self, model_name: str, num_layers_to_finetune: int, **hf_kwargs):
        super().__init__()
        default_hf_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True
        }
        default_hf_kwargs.update(hf_kwargs)
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **default_hf_kwargs)
        self.model.gradient_checkpointing_enable()

        # 1. Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. Unfreeze the top N layers if specified
        if num_layers_to_finetune > 0 and hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            num_layers = len(self.model.model.layers)
            for i in range(num_layers - num_layers_to_finetune, num_layers):
                for param in self.model.model.layers[i].parameters():
                    param.requires_grad = True

        # 3. Replace the head and ensure it's trainable
        hidden_size = self.model.config.hidden_size
        self.model.lm_head = nn.Linear(hidden_size, 1, bias=False)
        for param in self.model.lm_head.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None):
        """
        Performs a forward pass and returns the logits for the last token.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Return the logits for the last token in the sequence for classification
        return outputs.logits[:, -1, :]


class BaseCausalLMFromScratch(nn.Module):
    """
    An abstract base class for creating causal language models from scratch for binary classification.
    
    This class handles the common forward pass logic. Subclasses are responsible for
    creating the specific model configuration and initializing the model architecture.
    """
    def __init__(self):
        super().__init__()
        self.model = None # Subclasses must initialize this

    def _create_config(self, model_name_for_config: str, settings, overrides: Dict[str, Any] = None):
        """Helper to create a model config, overriding defaults with settings."""
        config = AutoConfig.from_pretrained(model_name_for_config)
        config.vocab_size = settings.vocab_size
        config.max_position_embeddings = settings.context_length
        config.bos_token_id = settings.BOS_TOKEN
        config.use_cache = False
        
        if overrides:
            for key, value in overrides.items():
                setattr(config, key, value)
        return config

    def forward(self, x, attention_mask=None):
        """Performs a forward pass and returns the logits for the last token."""
        if self.model is None:
            raise NotImplementedError("Subclasses of BaseCausalLMFromScratch must initialize self.model.")
        output = self.model(input_ids=x, attention_mask=attention_mask)
        return output.logits[:, -1, :] # Return last token's logit

# =============================================================================
# Fine-tuning Model Implementations
# =============================================================================

class FinetuneLlama3Coder(BaseFinetuningModel):
    def __init__(self, num_layers_to_finetune: int):
        super().__init__(model_name="meta-llama/Llama-3.2-1B", num_layers_to_finetune=num_layers_to_finetune)

class FinetuneQwen3Coder(BaseFinetuningModel):
    def __init__(self, num_layers_to_finetune: int):
        super().__init__(model_name="Qwen/Qwen3-1.7B", num_layers_to_finetune=num_layers_to_finetune)

class FinetuneDeepSeekCoder(BaseFinetuningModel):
    def __init__(self, num_layers_to_finetune: int):
        super().__init__(model_name="deepseek-ai/deepseek-coder-1.3b-base", num_layers_to_finetune=num_layers_to_finetune)

# =============================================================================
# From-Scratch Model Implementations
# =============================================================================

class Llama3(BaseCausalLMFromScratch):
    def __init__(self, settings):
        super().__init__()
        config = self._create_config("meta-llama/Llama-3.2-1B", settings)
        self.model = LlamaForCausalLM(config)
        self.model.gradient_checkpointing_enable()
        self.model.lm_head = nn.Linear(config.hidden_size, 1, bias=False)

class DeepSeek(BaseCausalLMFromScratch):
    def __init__(self, settings):
        super().__init__()
        config = self._create_config("deepseek-ai/deepseek-coder-1.3b-base", settings)
        self.model = AutoModelForCausalLM.from_config(config)
        self.model.gradient_checkpointing_enable()
        self.model.lm_head = nn.Linear(config.hidden_size, 1, bias=False)

class Qwen3(BaseCausalLMFromScratch):
    def __init__(self, settings):
        super().__init__()
        # Determine the correct base model for config based on settings.model string
        if settings.model == "qwen0.6B":
            model_name_for_config = "Qwen/Qwen3-0.6B"
        else: # Default to 1.7B for "qwen1.7B" or other cases
            model_name_for_config = "Qwen/Qwen3-1.7B"

        config = self._create_config(model_name_for_config, settings)
        self.model = AutoModelForCausalLM.from_config(config) # Use AutoModel for flexibility
        self.model.gradient_checkpointing_enable()
        self.model.lm_head = nn.Linear(config.hidden_size, 1, bias=True)

# =============================================================================
# Bloom
# =============================================================================

class EnsembleBLOOM(nn.Module):
    def __init__(self, config, num_models):
        super().__init__()
        self.models = nn.ModuleList([BloomModel(config) for _ in range(num_models)])
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, x, attention_mask=None):
        hidden_states = [model(input_ids=x, attention_mask=attention_mask).last_hidden_state[:, -1, :] for model in self.models]
        pooled_output = torch.mean(torch.stack(hidden_states, dim=0), dim=0)
        return self.classifier(pooled_output)

# =============================================================================
# Model Factory using a Registry
# =============================================================================

# The MODEL_REGISTRY maps the model name from settings.py to the corresponding class.
MODEL_REGISTRY = {
    # Fine-tuning models
    "llama3_finetune": FinetuneLlama3Coder,
    "qwen3_finetune": FinetuneQwen3Coder,
    "deepseek_finetune": FinetuneDeepSeekCoder,
    # From-scratch models
    "llama3": Llama3,
    "deepseek": DeepSeek,
    "qwen1.7B": Qwen3,
    "qwen0.6B": Qwen3,
    # Ensemble models
    "bloom": EnsembleBLOOM,
}

def get_model(settings) -> nn.Module:
    """
    Factory function to create and return a model based on settings.

    This function uses a registry (`MODEL_REGISTRY`) to find the appropriate
    model class and then instantiates it with the required parameters from the
    settings object. This approach is easily extensible.
    
    Args:
        settings: A configuration object with attributes like 'model', 'vocab_size', etc.

    Returns:
        An initialized torch.nn.Module instance.
    """
    model_name = settings.model
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"{model_name} is no longer supported. Please use a Qwen3 model.")

    model_class = MODEL_REGISTRY[model_name]

    # Instantiate the model with the correct arguments
    if "finetune" in model_name:
        model = model_class(num_layers_to_finetune=settings.num_layers_to_finetune)
        return model
    
    elif model_name == "bloom":
        config = BloomConfig(
            vocab_size=settings.vocab_size,
            n_layer=settings.n_layer,
            n_head=settings.n_head,
            hidden_size=settings.n_embd,
        )
        model = EnsembleBLOOM(config, settings.num_models)
    
    else:
        model = model_class(settings)

    return model.to(settings.device)