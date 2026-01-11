# SAM Audio Serverless Worker for RunPod

This repository contains a RunPod serverless worker for [SAM Audio](https://github.com/facebookresearch/sam-audio). 

It implements the **Full** version of the model, supporting all modalities including text descriptions and temporal anchors (span prompting).

[![Runpod](https://api.runpod.io/badge/Jourdelune/sam-audio)](https://console.runpod.io/hub/Jourdelune/sam-audio)

## Features

- **Text Prompting**: Isolate sounds using natural language descriptions.
- **Span Prompting**: Isolate sounds occurring at specific time intervals (anchors).
- **High Quality**: Uses full vision encoders and rankers for best performance.
- **Audio Chunking**: Automatically processes long audio files in chunks (text-prompting only).
- **Serverless Ready**: Compliant with RunPod's serverless endpoint API.

## Input Format

The worker expects a JSON payload with the following structure:

### Text Prompting
```json
{
  "input": {
    "audio_url": "https://example.com/path/to/audio_file.wav",
    "description": "A dog barking"
  }
}
```

### Span Prompting (Anchors)
Isolate sounds based on time ranges. An anchor is defined as `[type, start_time, end_time]`.
- `"+"`: Positive anchor (include sound in this range).
- `"-"`: Negative anchor (exclude sound in this range).

```json
{
  "input": {
    "audio_url": "https://example.com/path/to/audio_file.wav",
    "anchors": [
      ["+", 1.5, 3.0],
      ["-", 0.0, 1.0]
    ]
  }
}
```

## Output Format

The worker returns a JSON object with base64-encoded audio files:

```json
{
  "target_audio": "base64_encoded_wav_string...",
  "residual_audio": "base64_encoded_wav_string...",
  "status": "success"
}
```

- `target_audio`: The isolated sound matching the description/anchors.
- `residual_audio`: The background audio with the target removed.
