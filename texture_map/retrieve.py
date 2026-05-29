import argparse
import json
from pathlib import Path

import numpy as np
from msclap import CLAP


SUPPORTED_MODEL = "msclap"


def load_index(index_dir: Path) -> tuple[dict, np.ndarray]:
    metadata_path = index_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json missing: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    embeddings_file = metadata.get("embeddings_file", "embeddings.npy")
    embeddings_path = index_dir / embeddings_file

    if not embeddings_path.exists():
        raise FileNotFoundError(f"embeddings.npy missing: {embeddings_path}")

    embeddings = np.load(embeddings_path)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape: {embeddings.shape}")

    samples = metadata.get("samples")
    if not isinstance(samples, list):
        raise ValueError("metadata.json is missing a valid 'samples' list.")

    num_samples = metadata.get("num_samples")
    if num_samples != len(samples):
        raise ValueError(
            f"metadata num_samples ({num_samples}) does not match samples length ({len(samples)})."
        )

    if embeddings.shape[0] != num_samples:
        raise ValueError(
            f"embedding count ({embeddings.shape[0]}) does not match num_samples ({num_samples})."
        )

    return metadata, embeddings


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vec)

    if norm < 1e-12:
        raise ValueError("Text embedding has near-zero norm and cannot be normalized.")

    return vec / norm


def embed_text(prompt: str, model_name: str, model_version: str) -> np.ndarray:
    if model_name != SUPPORTED_MODEL:
        raise ValueError(f"Unsupported embedding model: {model_name}")

    print(f"Loading CLAP model: {model_name} version={model_version}")
    clap_model = CLAP(version=model_version, use_cuda=False)

    print("Generating text embedding...")
    text_embedding = clap_model.get_text_embeddings([prompt])
    return normalize_vector(text_embedding)


def retrieve(prompt: str, index_dir: Path, k: int) -> list[dict]:
    metadata, audio_embeddings = load_index(index_dir)

    model_name = metadata.get("embedding_model")
    model_version = metadata.get("embedding_model_version")

    text_embedding = embed_text(prompt, model_name, model_version)

    if audio_embeddings.shape[1] != text_embedding.shape[0]:
        raise ValueError(
            "Text embedding dimension "
            f"({text_embedding.shape[0]}) does not match audio embedding dimension "
            f"({audio_embeddings.shape[1]})."
        )

    scores = audio_embeddings @ text_embedding
    result_count = min(k, len(metadata["samples"]))
    top_indices = np.argsort(scores)[::-1][:result_count]

    results = []
    for index in top_indices:
        sample = metadata["samples"][int(index)]
        results.append(
            {
                "sample": sample,
                "score": float(scores[index]),
            }
        )

    return results


def print_results(prompt: str, results: list[dict]) -> None:
    print()
    print(f'Prompt: "{prompt}"')
    print()
    print("Top matches:")

    if not results:
        print("No samples found.")
        return

    names = [
        Path(result["sample"].get("path", result["sample"].get("id", "unknown"))).name
        for result in results
    ]
    name_width = max(len(name) for name in names)

    for rank, (result, name) in enumerate(zip(results, names), start=1):
        score = result["score"]
        print(f"{rank}. {name:<{name_width}}    score: {score:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve TextureMap samples that match a text prompt."
    )
    parser.add_argument("prompt", help="Text prompt to search for.")
    parser.add_argument("index_dir", type=Path, help="Directory containing index files.")
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of matches to return. Defaults to 8.",
    )

    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be at least 1.")

    try:
        results = retrieve(args.prompt, args.index_dir, args.k)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print_results(args.prompt, results)


if __name__ == "__main__":
    main()
