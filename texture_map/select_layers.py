import argparse
from pathlib import Path

import librosa
import numpy as np

from texture_map.retrieve import retrieve


EPSILON = 1e-8


def resolve_sample_path(sample: dict, index_dir: Path) -> Path:
    sample_path = Path(sample["path"])

    if sample_path.is_absolute() or sample_path.exists():
        return sample_path

    project_relative_path = index_dir.parent / sample_path
    if project_relative_path.exists():
        return project_relative_path

    return sample_path


def extract_features(audio_path: Path, sample: dict) -> dict:
    audio, sample_rate = librosa.load(audio_path, sr=None, mono=True)

    if audio.size == 0:
        raise ValueError("audio file is empty")
    
    duration = float(sample.get("duration") or librosa.get_duration(y=audio, sr=sample_rate))
    rms = librosa.feature.rms(y=audio)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)[0]
    onset_envelope = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        units="frames",
    )

    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    return {
        "duration": duration,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "rms_variation_ratio": float(rms_std / (rms_mean + EPSILON)),
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
        "zero_crossing_rate_mean": float(np.mean(zero_crossing_rate)),
        "onset_strength_mean": float(np.mean(onset_envelope)) if onset_envelope.size else 0.0,
        "onset_count": int(len(onset_frames)),
    }


def classify_roles(features: dict) -> list[tuple[str, str]]:
    duration = features["duration"]
    rms_variation = features["rms_variation_ratio"]
    centroid = features["spectral_centroid_mean"]
    bandwidth = features["spectral_bandwidth_mean"]
    zcr = features["zero_crossing_rate_mean"]
    onset_count = features["onset_count"]
    onset_density = onset_count / max(duration, EPSILON)

    roles = []

    if duration >= 6.0 and rms_variation <= 0.65:
        roles.append(("bed", "long + steady"))

    if duration <= 3.0 and (onset_density >= 0.75 or rms_variation >= 0.9):
        roles.append(("foreground", "short/transient"))

    if duration >= 1.5 and centroid <= 450.0:
        roles.append(("low", "low spectral centroid"))

    if centroid >= 3500.0 or zcr >= 0.12:
        roles.append(("detail", "bright/high centroid"))

    if duration >= 2.0 and (rms_variation >= 0.9 or bandwidth >= 3000.0):
        roles.append(("motion", "variable/noisy texture"))

    if not roles or bandwidth >= 2200.0 or zcr >= 0.08:
        roles.append(("texture", "noisy/grainy texture"))

    return roles


def analyze_candidate(result: dict, index_dir: Path) -> dict | None:
    sample = result["sample"]
    audio_path = resolve_sample_path(sample, index_dir)

    if not audio_path.exists():
        print(f"Warning: skipping missing audio file: {audio_path}")
        return None

    try:
        features = extract_features(audio_path, sample)
    except Exception as exc:
        print(f"Warning: skipping {audio_path}: {exc}")
        return None

    roles = classify_roles(features)

    return {
        "sample": sample,
        "score": result["score"],
        "path": audio_path,
        "features": features,
        "roles": roles,
    }


def add_best_for_role(
    selected: list[dict],
    candidates: list[dict],
    used_ids: set[str],
    role: str,
    limit: int,
    target_layers: int,
) -> None:
    for candidate in candidates:
        if len(selected) >= target_layers or limit <= 0:
            return

        sample_id = sample_key(candidate["sample"])
        if sample_id in used_ids:
            continue

        reason = next(
            (reason for candidate_role, reason in candidate["roles"] if candidate_role == role),
            None,
        )
        if reason is None:
            continue

        selected.append({**candidate, "selected_role": role, "reason": reason})
        used_ids.add(sample_id)
        limit -= 1


def sample_key(sample: dict) -> str:
    return str(sample.get("id") or sample.get("path"))


def choose_balanced_layers(candidates: list[dict], layers: int) -> list[dict]:
    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)

    selected = []
    used_ids = set()

    role_plan = [
        ("bed", 2),
        ("low", 1),
        ("detail", 1),
        ("texture", 1),
        ("foreground", 2),
        ("motion", 1),
    ]

    for role, limit in role_plan:
        add_best_for_role(selected, candidates, used_ids, role, limit, layers)

    for candidate in candidates:
        if len(selected) >= layers:
            break

        sample_id = sample_key(candidate["sample"])
        if sample_id in used_ids:
            continue

        role, reason = candidate["roles"][0]
        selected.append({**candidate, "selected_role": role, "reason": reason})
        used_ids.add(sample_id)

    return selected


def analyze_retrieved_candidates(results: list[dict], index_dir: Path) -> list[dict]:
    candidates = []

    for result in results:
        candidate = analyze_candidate(result, index_dir)
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def select_layers(prompt: str, index_dir: Path, pool_k: int, layers: int) -> list[dict]:
    if pool_k < 1:
        raise ValueError("pool_k must be at least 1.")

    if layers < 1:
        raise ValueError("layers must be at least 1.")

    retrieved = retrieve(prompt, index_dir, pool_k)
    candidates = analyze_retrieved_candidates(retrieved, index_dir)
    return choose_balanced_layers(candidates, layers)


def print_selected_layers(
    prompt: str,
    selected: list[dict],
    pool_size: int,
    target_layers: int,
) -> None:
    print()
    print(f'Prompt: "{prompt}"')
    print()
    print(f"Candidate pool size: {pool_size}")
    print(f"Target layers: {target_layers}")
    print()

    if not selected:
        print("No valid candidates remained after feature analysis.")
        return

    print("Selected layers:")
    print()

    names = [
        Path(layer["sample"].get("path", layer["sample"].get("id", "unknown"))).name
        for layer in selected
    ]
    role_width = max(len(layer["selected_role"]) for layer in selected)
    name_width = max(len(name) for name in names)

    for layer, name in zip(selected, names):
        role = layer["selected_role"]
        score = layer["score"]
        reason = layer["reason"]
        print(
            f"[{role:<{role_width}}] {name:<{name_width}}    "
            f"score: {score:.2f}    reason: {reason}"
        )

    print()
    print("Recipe:")
    print()

    role_phrases = {
        "bed": "Use {name} as a background bed.",
        "low": "Add {name} for low pressure.",
        "detail": "Place {name} as bright detail.",
        "texture": "Layer {name} as texture.",
        "foreground": "Use {name} as a foreground accent.",
        "motion": "Use {name} as motion/noise texture.",
    }

    bed_count = 0
    for layer, name in zip(selected, names):
        role = layer["selected_role"]

        if role == "bed":
            bed_count += 1
            if bed_count > 1:
                print(f"Layer {name} quietly underneath.")
                continue

        print(role_phrases.get(role, "Layer {name} into the texture.").format(name=name))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a balanced set of TextureMap layers from prompt retrieval."
    )
    parser.add_argument("prompt", help="Text prompt to search for.")
    parser.add_argument("index_dir", type=Path, help="Directory containing index files.")
    parser.add_argument(
        "--pool-k",
        type=int,
        default=20,
        help="Number of retrieved candidates to analyze. Defaults to 20.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=6,
        help="Target number of layers to select. Defaults to 6.",
    )

    args = parser.parse_args()

    if args.pool_k < 1:
        parser.error("--pool-k must be at least 1.")

    if args.layers < 1:
        parser.error("--layers must be at least 1.")

    try:
        retrieved = retrieve(args.prompt, args.index_dir, args.pool_k)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    candidates = analyze_retrieved_candidates(retrieved, args.index_dir)
    selected = choose_balanced_layers(candidates, args.layers)

    print_selected_layers(
        prompt=args.prompt,
        selected=selected,
        pool_size=len(retrieved),
        target_layers=args.layers,
    )


if __name__ == "__main__":
    main()
