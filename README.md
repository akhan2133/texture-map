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

## Stage 3: Prompt Based Retrieval (Implemented)
- Loads index/metadata.json
- Loads index/embeddings.npy
- Embeds a text prompt using the same CLAP model version as the index
- Normalizes the text embedding
- Compares the prompt embedding to audio embeddings using dot product
- Prints the top matching samples with similarity scores

## Stage 4: Layer Selection (Implemented)
- Retrieves a larger candidate pool from the prompt
- Analyzes simple audio features for each candidate
- Assigns rough sound-design roles such as bed, low, detail, texture, foreground, and motion
- Selects a balanced set of layers instead of blindly taking the top matches
- Prints the selected layers with scores, role reasons, and a basic recipe

## Stage 5: Audio Transformation (Implemented)
- Loads one processed WAV file
- Keeps audio in stereo NumPy format
- Applies simple producer controls to one layer at a time
- Uses real Pedalboard effects for filters, reverb, delay, pitch shifting, distortion, gain, and limiting
- Supports brightness, distance, movement, tension, and texture controls
- Can trim or loop a clip to a target duration
- Exports one transformed WAV file

## Stage 6: Mix Rendering (Implemented)
- Takes one text prompt and renders one final soundscape WAV
- Reuses prompt retrieval and balanced layer selection
- Uses density only to choose the target layer count
- Transforms each selected layer with the producer controls
- Places the transformed clips on a fixed stereo timeline
- Applies safe master limiting/normalization
- Exports a recipe JSON explaining the rendered layers

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
# Disable Hugging Face's Xet downloader. 
# On some systems, Xet downloads can hang at 0% when fetching models such as MS-CLAP. 
# TextureMap does not require Xet specifically.
# So, this uses the standard Hugging Face download path to make setup more reliable.
export HF_HUB_DISABLE_XET=1
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

### Step 3
After building the index, retrieve sounds that match a text prompt:
```bash
python -m texture_map.retrieve "dark souls ui menu click sound" index/ --k 3
```

This prints the top matching samples:
```text
Prompt: "dark souls ui menu click sound"

Top matches:
1. ui_1_mystical_selection.wav    score: 0.53
2. ui_5_confirmation.wav          score: 0.51
3. texture_1_sweep.wav            score: 0.50
```

The --k flag controls how many matches are shown. It is optional and defaults to 8 if not provided. The value must be a whole number of at least 1. If --k is larger than the number of indexed samples, TextureMap will simply show all available samples.

### Step 4
After building the index, select a balanced set of layers from a larger retrieval pool:
```bash
python -m texture_map.select_layers "<your-prompt>" index/ --pool-k 20 --layers 6
```

This prints the selected layer roles and a simple recipe:
```text
Prompt: "insanely pretty vox"

Candidate pool size: 20
Target layers: 6

Selected layers:

[bed       ] texture_5_space_fantasy.wav       score: 0.13    reason: long + steady
[bed       ] ambience_7_dark_mystical.wav      score: 0.12    reason: long + steady
[detail    ] music_9_good_morning_tokyo.wav    score: 0.09    reason: bright/high centroid
[texture   ] ui_10_chirp.wav                   score: 0.13    reason: noisy/grainy texture
[foreground] impact_2_man_hit.wav              score: 0.07    reason: short/transient
[motion    ] music_8_big_bad_boss.wav          score: 0.10    reason: variable/noisy texture

Recipe:

Use texture_5_space_fantasy.wav as a background bed.
Layer ambience_7_dark_mystical.wav quietly underneath.
Place music_9_good_morning_tokyo.wav as bright detail.
Layer ui_10_chirp.wav as texture.
Use impact_2_man_hit.wav as a foreground accent.
Use music_8_big_bad_boss.wav as motion/noise texture.
```

The --pool-k flag controls how many retrieval matches are analyzed before layer selection. The --layers flag controls the target number of selected layers. Both values must be whole numbers of at least 1.

### Step 5
After selecting or choosing a single processed WAV, transform it with producer controls:
```bash
python -m texture_map.transform samples/processed/ambience_5_rain.wav outputs/test_transform.wav \
  --brightness 0.7 \
  --distance 0.4 \
  --movement 0.5 \
  --tension 0.3 \
  --texture 0.6 \
  --duration 8 \
  --normalize
```

This loads one input WAV, applies the controls, and writes one transformed WAV:
```text
Transformed one TextureMap layer.
Input: samples/processed/ambience_5_rain.wav
Output: outputs/test_transform.wav
Sample rate: 44100 Hz
Duration: 8.00 seconds
Controls:
  brightness: 0.70
  distance: 0.40
  movement: 0.50
  tension: 0.30
  texture: 0.60
Target duration: 8.00 seconds
Normalize before save: True
```

The controls are floating point values from 0.0 to 1.0:

- `--brightness`: 0.0 is darker and low-passed, 0.5 is mostly unchanged, and 1.0 is brighter.
- `--distance`: 0.0 is close and present, while 1.0 is quieter, darker, narrower, and wetter.
- `--movement`: 0.0 is static, while 1.0 adds a more obvious slow autopan.
- `--tension`: 0.0 is smoother, while 1.0 is sharper and more saturated.
- `--texture`: 0.0 is clean, while 1.0 is more delayed, smeared, and reverse-blended.

The `--duration` flag trims or loops the source to the target length before transformation. The `--normalize` flag peak-normalizes the result before saving. Without `--normalize`, TextureMap still keeps the saved WAV at a safe output level.

### Step 6
After building the index, render a complete prompt-based soundscape:
```bash
python -m texture_map.mix "wet neon alley" index/ outputs/wet_neon_alley.wav --duration 20 --seed 42
```

TextureMap retrieves candidates, selects a balanced layer palette, transforms each layer, places clips on a fixed timeline, and writes:
```text
outputs/wet_neon_alley.wav
outputs/wet_neon_alley.recipe.json
```

Use the producer controls to shape the rendered mix:
```bash
python -m texture_map.mix "wet neon alley" index/ outputs/wet_neon_alley.wav \
  --duration 20 \
  --seed 42 \
  --pool-k 20 \
  --brightness 0.7 \
  --density 0.6 \
  --distance 0.4 \
  --movement 0.5 \
  --tension 0.3 \
  --texture 0.6 \
  --normalize
```

Density maps to the number of layers requested from Stage 4: `0.0` selects 2 layers, `0.5` selects 5 layers, and `1.0` selects 8 layers. It does not select layers a second time.
   
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
    retrieve.py
    select_layers.py
    transform.py
    mix.py
```
