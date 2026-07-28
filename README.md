# TextureMap

TextureMap is a prompt-driven sound-design sketchpad. It searches an indexed sample
library, proposes a balanced set of layers, lets you preview and refine each source
region, then transforms and mixes the curated material into a finished WAV. Audio is
retrieved from your own library rather than generated from scratch.

## Demo

[Watch the TextureMap prototype demo on YouTube](https://youtu.be/U5bWId2d440)

The demo shows the full Streamlit workflow: entering a prompt, reviewing suggested samples, selecting source regions, adjusting producer controls, generating the final sound, and exporting the WAV and recipe JSON.

## How It Works

1. Raw samples are standardized as stereo, 44.1 kHz WAV files with consistent
   loudness and a maximum duration of 20 seconds.
2. MS-CLAP converts the processed library into searchable audio embeddings.
3. A text prompt retrieves related sounds, and lightweight audio analysis balances
   them across roles such as bed, texture, detail, foreground, low, and motion.
4. In the Streamlit app, you can preview every proposal, exclude unwanted layers,
   and drag or resize the purple waveform region to choose the exact source audio.
5. Producer controls transform the curated layers before they are placed on a stereo
   timeline and safely limited.
6. TextureMap exports the final WAV and a JSON recipe containing the prompt,
   controls, source regions, placements, gains, and selected clips.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Prepare a Sample Library

Place supported audio files (`.wav`, `.mp3`, `.flac`, `.aiff`, `.aif`, or `.ogg`)
in `samples/raw/`, then standardize them:

```bash
python -m texture_map.prepare samples/raw samples/processed
```

This creates processed WAV files and `samples/processed/metadata.json`. Build the
search index from that metadata:

```bash
export HF_HUB_DISABLE_XET=1
python -m texture_map.embed samples/processed/metadata.json index/
```

The environment variable uses Hugging Face's standard download path, which avoids
Xet download stalls seen on some systems. Embedding runs on CPU by default; add
`--cuda` when CUDA is available.

A complete index contains:

```text
index/
├── embeddings.npy
└── metadata.json
```

## Run the App

Once the sample library and index are ready, launch the interface from the repository
root:

```bash
streamlit run app.py
```

The app uses a discovery-and-curation workflow:

1. Enter a sound-design prompt, choose a proposal count from 2–10, set the retrieval
   pool size, and click **Find Layers**.
2. Use each compact source player to browse its selected region and use **Include**
   to control whether the layer reaches the mix.
3. Expand a layer to inspect its role, similarity, duration, and selection reason.
   Drag the purple waveform region to move it or drag either edge to resize it. When
   the edit ends, the compact player updates to preview only that region.
4. Set brightness, distance, movement, tension, texture, output duration, and seed.
5. Click **Generate Texture** to render, preview, and download the WAV and recipe
   JSON.

The requested proposal count is not a required final layer count. Excluded layers are
not rendered, and every included waveform region is cropped before transformation,
looping, and timeline placement.

## Command-Line Tools

The same pipeline can be used without Streamlit.

Retrieve the closest prompt matches:

```bash
python -m texture_map.retrieve "dark souls UI menu click" index/ --k 8
```

Select a role-balanced palette from a larger retrieval pool:

```bash
python -m texture_map.select_layers "wet neon alley" index/ --pool-k 20 --layers 6
```

Transform a single processed WAV:

```bash
python -m texture_map.transform \
  samples/processed/ambience_5_rain.wav \
  outputs/rain_transformed.wav \
  --brightness 0.7 \
  --distance 0.4 \
  --movement 0.5 \
  --tension 0.3 \
  --texture 0.6 \
  --duration 8 \
  --normalize
```

Render a complete prompt-driven mix automatically:

```bash
python -m texture_map.mix \
  "wet neon alley" \
  index/ \
  outputs/wet_neon_alley.wav \
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

All producer controls accept values from `0.0` to `1.0`:

| Control | Effect |
| --- | --- |
| `brightness` | Darker and filtered to brighter and more present |
| `density` | Requests 2–10 layers for automatic CLI rendering |
| `distance` | Close and dry to quieter, darker, narrower, and wetter |
| `movement` | Static positioning to stronger slow stereo motion |
| `tension` | Smooth processing to sharper saturation |
| `texture` | Clean audio to delayed, smeared, and reverse-blended audio |

For automatic CLI mixes, density `0.0`, `0.5`, and `1.0` request 2, 6, and 10
layers respectively. In the app, the explicitly curated layer set takes precedence.
The seed makes randomized gains and timeline placement reproducible.

## Outputs

Each complete render writes two files:

```text
outputs/<name>.wav
outputs/<name>.recipe.json
```

The recipe records the rendered layer count, source filenames and regions, assigned
roles, similarity scores, timeline positions, gains, producer controls, duration,
sample rate, and seed.

## Project Structure

```text
texture-map/
├── app.py                         # Streamlit discovery, curation, and render UI
├── README.md
├── requirements.txt
├── samples/
│   ├── raw/                       # User-supplied source audio
│   └── processed/                 # Standardized WAVs and metadata.json
├── index/
│   ├── embeddings.npy             # Normalized MS-CLAP audio embeddings
│   └── metadata.json              # Search-index metadata
├── outputs/                       # Rendered WAVs and recipe JSON files
└── texture_map/
    ├── __init__.py
    ├── prepare.py                 # Sample loading and standardization
    ├── embed.py                   # Audio embedding and index creation
    ├── retrieve.py                # Prompt embedding and similarity search
    ├── select_layers.py           # Feature analysis and balanced role selection
    ├── transform.py               # Per-layer producer-control processing
    ├── mix.py                     # Source cropping, placement, mixing, and recipes
    ├── waveform.py                # Streamlit bridge for editable waveform regions
    └── _wavesurfer_frontend/
        ├── index.html
        ├── assets/                # Bundled waveform component JavaScript and CSS
        └── NOTICE.md              # Upstream component attribution
```

`samples/`, `index/`, and `outputs/` contain local or generated data and are excluded
from version control by default.
