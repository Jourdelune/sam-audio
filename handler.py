import torch
import runpod
import torchaudio
import base64
import io
import os
import requests
import traceback
from utils import load_model, show_gpu_memory

# Global model variables
MODEL = None
PROCESSOR = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Use BF16 as requested in the optimization script
USE_BF16 = True
DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

def init_model():
    global MODEL, PROCESSOR
    if MODEL is None:
        print("--- Initializing Model ---")
        show_gpu_memory("Before loading model")
        
        # Get configuration from env
        model_name = os.environ.get("MODEL_NAME", "facebook/sam-audio-base")
        hf_token = os.environ.get("HF_TOKEN") or None
        
        try:
            MODEL, PROCESSOR = load_model(model_name, token=hf_token)
            MODEL = MODEL.eval().to(DEVICE, DTYPE)
            show_gpu_memory("After loading model")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

def download_audio(url):
    """Download audio from a URL into a BytesIO object"""
    print(f"Downloading audio from {url}...")
    headers = {'User-Agent': 'RunPod-Worker/1.0'}
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    return io.BytesIO(response.content)

def audio_to_base64(waveform, sample_rate):
    """Convert audio tensor to base64 encoded WAV"""
    buffer = io.BytesIO()
    torchaudio.save(buffer, waveform, sample_rate, format="wav")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def process_audio(audio_source, description, anchors=None, monitor_spans=False):
    """
    Process audio using the full model logic.
    Supports text (description) and span (anchors) prompting.
    """
    # Logic adapted from the provided script
    sample_rate = PROCESSOR.audio_sampling_rate
    
    # torchaudio.load handles file-like objects (BytesIO)
    audio, orig_sr = torchaudio.load(audio_source)
    
    # Resample if needed
    if orig_sr != sample_rate:
        print(f"Resampling from {orig_sr} to {sample_rate}")
        resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
        audio = resampler(audio)
    
    # Convert to mono if needed
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
        
    audio_duration = audio.shape[1] / sample_rate
    print(f"Audio duration: {audio_duration:.2f}s")
    
    CHUNK_DURATION = 25.0
    MAX_CHUNK_SAMPLES = int(sample_rate * CHUNK_DURATION)
    
    target_audio = None
    residual_audio = None
    
    # Check if we should use chunking
    # Note: Chunking with anchors is complicated because anchors are time-relative.
    # For simplicity, if anchors are present, we disable chunking or warn.
    if audio.shape[1] > MAX_CHUNK_SAMPLES and not anchors:
        print(f"Audio is {audio_duration:.1f}s, using chunking ({CHUNK_DURATION}s chunks)")
        
        # Split audio into chunks
        audio_tensor = audio.squeeze(0).to(DEVICE, DTYPE)
        chunks = torch.split(audio_tensor, MAX_CHUNK_SAMPLES, dim=-1)
        
        out_target = []
        out_residual = []
        
        for i, chunk in enumerate(chunks):
            # Skip very short chunks (< 1s)
            if chunk.shape[-1] < sample_rate:
                continue
                
            batch = PROCESSOR(
                audios=[chunk.unsqueeze(0)],
                descriptions=[description]
                # Anchors not supported in chunking mode yet
            ).to(DEVICE)
            
            with torch.inference_mode():
                # Use autocast if on CUDA
                with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                    result = MODEL.separate(
                        batch,
                        predict_spans=monitor_spans,
                        reranking_candidates=3 # Increased for full model
                    )
            
            out_target.append(result.target[0].cpu())
            out_residual.append(result.residual[0].cpu())
            
            # Memory cleanup
            del batch, result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        target_audio = torch.cat(out_target, dim=-1).clamp(-1, 1).float().unsqueeze(0)
        residual_audio = torch.cat(out_residual, dim=-1).clamp(-1, 1).float().unsqueeze(0)
        
    else:
        if audio.shape[1] > MAX_CHUNK_SAMPLES and anchors:
            print("Warning: Audio is long but anchors are provided. Processing as single batch (may OOM).")
            
        print("Processing as single batch")
        
        # Prepare inputs
        inputs_kwargs = {
            "audios": [audio],
            "descriptions": [description] if description else None,
        }
        
        if anchors:
             # anchors format expected by processor: list of lists of anchors for each audio
             # Each anchor is [type, start, end] e.g., ["+", 1.0, 2.0]
             # If passed via JSON, ensure format is correct
             inputs_kwargs["anchors"] = [anchors]
             
        # If no description provided but anchors are, descriptions needs to be [""]?
        # Processor handles it usually, but let's be safe
        if not description and anchors:
             inputs_kwargs["descriptions"] = [""]

        batch = PROCESSOR(**inputs_kwargs).to(DEVICE)
         
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda"), dtype=DTYPE):
                result = MODEL.separate(
                    batch, 
                    predict_spans=monitor_spans,
                    reranking_candidates=5 # Use higher candidate count for full model quality
                )
        
        target_audio = result.target[0].float().unsqueeze(0).cpu()
        residual_audio = result.residual[0].float().unsqueeze(0).cpu()
        
        del batch, result

    return target_audio, residual_audio, sample_rate

def handler(job):
    """
    RunPod Handler function
    """
    job_input = job['input']
    
    # Inputs
    audio_url = job_input.get('audio_url')
    description = job_input.get('description', "") 
    anchors = job_input.get('anchors', None)
    
    # Validation
    if not audio_url:
        return {"error": "Missing 'audio_url' in input."}
    if not description and not anchors:
        return {"error": "Provide either 'description' or 'anchors' (or both)."}
    
    try:
        # Download
        audio_file = download_audio(audio_url)
        
        # Process
        target, residual, sr = process_audio(audio_file, description, anchors=anchors)
        
        # Encode results
        target_b64 = audio_to_base64(target, sr)
        residual_b64 = audio_to_base64(residual, sr)
        
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return {
            "target_audio": target_b64,
            "residual_audio": residual_b64,
            "status": "success"
        }
        
    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    init_model()
    runpod.serverless.start({"handler": handler})
