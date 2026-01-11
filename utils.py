import torch
import gc
import types

def show_gpu_memory(label: str = ""):
    """Show complete GPU memory stats (matches nvidia-smi more closely)"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[GPU Memory{' - ' + label if label else ''}] "
              f"Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Peak: {max_allocated:.2f}GB")

def load_model(model_name: str = "facebook/sam-audio-base", token: str = None):
    """
    Load the full SAM Audio model with all modalities enabled.
    
    Returns:
        model: SAM Audio model
        processor: SAM Audio processor
    """
    from sam_audio import SAMAudio, SAMAudioProcessor
    
    print(f"Loading {model_name}...")
    
    # Load model
    if token:
        model = SAMAudio.from_pretrained(model_name, token=token)
    else:
        model = SAMAudio.from_pretrained(model_name)
    
    processor = SAMAudioProcessor.from_pretrained(model_name)
    
    print("Model loaded successfully.")
    
    return model, processor
