import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import pyloudnorm as pyln


TARGET_SR = 44100
TARGET_LUFS = -18.4
MAX_SECONDS = 20.0


def load_audio(path: Path, target_sr: int = TARGET_SR):
    """
    Loads audio as stereo at the target sample rate.
    Shape returned: (samples, channels)
    """
    audio, _ = librosa.load(path, sr=target_sr, mono=False)

    # librosa returns:
    # mono: (samples,)
    # stereo: (channels, samples)

    if audio.ndim == 1:
        # mono -> stereo
        audio = np.stack([audio, audio], axis=0)

    if audio.shape[0] > 2:
        # weird multi-channel -> first two channels
        audio = audio[:2, :]

    if audio.shape[0] == 1:
        audio = np.vstack([audio, audio])

    # convert from (channels, samples) to (samples, channels) 
    audio = audio.T

    return audio, target_sr


def trim_audio(audio: np.ndarray, sr: int, max_seconds: float = MAX_SECONDS):
    max_samples = int(sr * max_seconds)

    if len(audio) > max_samples:
        audio = audio[:max_samples]

    return audio


def normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS):
    """
    Normalize perceived loudness to target LUFS.
    """
    meter = pyln.Meter(sr)

    loudness = meter.integrated_loudness(audio)

    # If the file is basically silent or weird, skip normalization
    if not np.isfinite(loudness):
        return audio, None

    normalized_audio = pyln.normalize.loudness(audio, loudness, target_lufs)

    # Prevent clipping after normalization
    peak = np.max(np.abs(normalized_audio))
    if peak > 1.0:
        normalized_audio = normalized_audio / peak * 0.98

    final_loudness = meter.integrated_loudness(normalized_audio)

    return normalized_audio, float(final_loudness)


def process_file(input_path: Path, output_dir: Path):
    audio, sr = load_audio(input_path)
    audio = trim_audio(audio, sr)
    audio, lufs = normalize_loudness(audio, sr)

    output_name = input_path.stem + ".wav"
    output_path = output_dir / output_name

    sf.write(output_path, audio, sr, subtype="PCM_16")

    duration = len(audio) / sr

    metadata = {
        "id": input_path.stem,
        "source_path": str(input_path),
        "path": str(output_path),
        "duration": round(duration, 3),
        "sample_rate": sr,
        "channels": audio.shape[1],
        "lufs": round(lufs, 2) if lufs is not None else None,
    }

    return metadata


def prepare_samples(raw_dir: str, processed_dir: str):
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)

    supported_extensions = [".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg"]

    audio_files = [
        path for path in raw_dir.iterdir()
        if path.suffix.lower() in supported_extensions
    ]

    all_metadata = []

    for audio_file in audio_files:
        print(f"Processing {audio_file.name}...")
        metadata = process_file(audio_file, processed_dir)
        all_metadata.append(metadata)

    metadata_path = processed_dir / "metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\nProcessed {len(all_metadata)} files.")
    print(f"Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare TextureMap audio samples.")
    parser.add_argument("raw_dir", help="Path to raw samples folder.")
    parser.add_argument("processed_dir", help="Path to processed samples folder.")

    args = parser.parse_args()

    prepare_samples(args.raw_dir, args.processed_dir)


if __name__ == "__main__":
    main()