import argparse
import json
from pathlib import Path

import numpy as np
from msclap import CLAP


MODEL_VERSION = "2023"


def load_metadata(metadata_path: Path) -> list[dict]:
    """
    Expected format of metadata:
    [
      {
        "id": "...",
        "source_path": "...",
        "path": "samples/processed/example.wav",
        "duration": ...,
        "sample_rate": ...,
        "channels": ...,
        "lufs": ...
      }
    ]
    """
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not isinstance(metadata, list):
        raise ValueError("Expected metadata.json to be a list of sample objects.")

    for entry in metadata:
        if not isinstance(entry, dict):
            raise ValueError(f"Expected each metadata entry to be an object: {entry}")

        if "id" not in entry:
            raise ValueError(f"Metadata entry is missing 'id': {entry}")

        if "path" not in entry:
            raise ValueError(f"Metadata entry is missing 'path': {entry}")

    return metadata


def resolve_audio_paths(metadata: list[dict]) -> list[Path]:
    """
    Convert each metadata entry's 'path' field into a real Path object.
    """
    audio_paths = []

    for entry in metadata:
        path = Path(entry["path"])

        if not path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {path}")

        if path.suffix.lower() != ".wav":
            raise ValueError(f"Expected a WAV file, got: {path}")

        audio_paths.append(path)

    return audio_paths


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize each embedding to length 1.

    This makes future cosine similarity easier:
    similarity = text_embedding @ audio_embedding
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms

def save_index(
    embeddings: np.ndarray,
    metadata: list[dict],
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    Save:
    - index/embeddings.npy
    - index/metadata.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "metadata.json"

    np.save(embeddings_path, embeddings)

    indexed_samples = []

    for i, entry in enumerate(metadata):
        indexed_entry = dict(entry)
        indexed_entry["embedding_row"] = i
        indexed_samples.append(indexed_entry)

    index_metadata = {
        "embedding_model": "msclap",
        "embedding_model_version": MODEL_VERSION,
        "embedding_dim": int(embeddings.shape[1]),
        "num_samples": int(embeddings.shape[0]),
        "embeddings_file": "embeddings.npy",
        "samples": indexed_samples,
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(index_metadata, f, indent=2)

    return embeddings_path, metadata_path


def main():
    parser = argparse.ArgumentParser(
        description="Build CLAP audio embeddings for processed TextureMap samples."
    )

    parser.add_argument(
        "metadata_json",
        type=Path,
        help="Path to samples/processed/metadata.json",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to write index files.",
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA GPU acceleration. Default is CPU.",
    )

    args = parser.parse_args()

    print(f"Reading metadata: {args.metadata_json}")
    metadata = load_metadata(args.metadata_json)

    print(f"Found {len(metadata)} samples.")

    audio_paths = resolve_audio_paths(metadata)

    print("Audio files to embed:")
    for path in audio_paths:
        print(f"  - {path}")

    print(f"Loading CLAP model: msclap version={MODEL_VERSION}")
    clap_model = CLAP(version=MODEL_VERSION, use_cuda=args.cuda)

    file_paths = [str(path) for path in audio_paths]

    print("Generating audio embeddings...")
    embeddings = clap_model.get_audio_embeddings(file_paths)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape: {embeddings.shape}")

    print(f"Raw embedding shape: {embeddings.shape}")

    print("Normalizing embeddings...")
    embeddings = normalize_embeddings(embeddings)

    embeddings_path, index_metadata_path = save_index(
        embeddings=embeddings,
        metadata=metadata,
        output_dir=args.output_dir,
    )

    print("Done.")
    print(f"Saved embeddings: {embeddings_path}")
    print(f"Saved metadata:   {index_metadata_path}")
    print(f"Embedding shape:  {embeddings.shape}")


if __name__ == "__main__":
    main()