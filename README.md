# TextureMap

TextureMap is a prompt-based sound design sketchpad.

The goal is to turn a folder of audio samples into layered sound textures based on a text prompt. Instead of generating audio from scratch, TextureMap uses retrieval, transformation, and mixing: it finds relevant sounds from a sample library, applies audio transformations, layers them together, and exports a final WAV file with a recipe of how it was made.

## Core Idea

```text
sample folder
   ↓
analyze + embed samples
   ↓
user prompt
   ↓
retrieve matching sounds
   ↓
choose 3–8 layers
   ↓
apply transformations
   ↓
mix soundscape
   ↓
export WAV + show recipe
```

This project is currently being built in stages.

## Stage 1: Load and Standardize Samples (Implemented)
- Scans a raw sample folder
- Loads supported audio files
- Converts audio to a standard format
- Resamples to 44.1kHz
- Converts mono files to stereo
- Trims long files to a maximum length
- Normalizes loudness to a target LUFS level
- Exports processed WAV files
- Writes a metadata.json file for later stages

## Stage 2: Embed Sample Library (Implemented)
- Reads samples/processed/metadata.json
- Loads each processed WAV file
- Uses CLAP through msclap to generate audio embeddings
- Normalizes embeddings for cosine similarity
- Saves embeddings to index/embeddings.npy
- Saves index metadata to index/metadata.json

## Stage 3: Prompt Based Retrieval (In Progress)
Use a text prompt to retrieve matching sounds from the embedded sample library.

## Installation:
Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage
### Step 1
Place raw audio files in:
```bash
samples/raw/
```

Then run:
```bash
python -m texture_map.prepare samples/raw samples/processed
```

Processed files and metadata will be written to:
```bash
samples/processed/
```

### Step 2
After generating processed WAVs and metadata, run:
```bash
python -m texture_map.embed samples/processed/metadata.json index/
```

This writes:
```bash
index/embeddings.npy
index/metadata.json
```

For GPU acceleration, you can optionally run:
```
python -m texture_map.embed samples/processed/metadata.json index/ --cuda
```
CPU is fine for smaller libraries.


## Project Structure
```text
texture_map/
  README.md
  requirements.txt
  index/
  samples/
    raw/
    processed/
  texture_map/
    __init__.py
    prepare.py
    embed.py
```