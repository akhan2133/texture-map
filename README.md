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

## Stage 2: Embed Sample Library (In Progress)
Generate audio embeddings for each processed sample so the system can search the library by meaning, mood, and texture.

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

## Project Structure
texture_map/
  README.md
  requirements.txt
  samples/
    raw/
    processed/
  texture_map/
    __init__.py
    prepare.py