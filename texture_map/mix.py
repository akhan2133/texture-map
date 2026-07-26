import argparse
import json
from pathlib import Path

import librosa
import numpy as np

from texture_map.select_layers import resolve_sample_path, select_layers
from texture_map.transform import (
    apply_fade,
    apply_gain,
    ensure_stereo,
    load_audio,
    normalize_peak,
    save_audio,
    transform_layer,
    trim_or_loop,
)


CONTROL_NAMES = (
    "brightness",
    "density",
    "distance",
    "movement",
    "tension",
    "texture",
)

ROLE_GAIN_RANGES = {
    "bed": (-10.0, -7.0),
    "low": (-9.0, -6.0),
    "texture": (-12.0, -7.0),
    "motion": (-12.0, -7.0),
    "foreground": (-9.0, -4.0),
    "detail": (-9.0, -4.0),
}


def clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def density_to_layer_count(
    density: float,
    min_layers: int = 2,
    max_layers: int = 8,
) -> int:
    if min_layers < 1:
        raise ValueError("min_layers must be at least 1.")

    if max_layers < min_layers:
        raise ValueError("max_layers must be greater than or equal to min_layers.")

    density = clamp01(density)
    return int(round(min_layers + density * (max_layers - min_layers)))


def default_controls() -> dict[str, float]:
    return {name: 0.5 for name in CONTROL_NAMES}


def _merge_controls(controls: dict[str, float] | None) -> dict[str, float]:
    merged = default_controls()

    if controls:
        for name, value in controls.items():
            if name in merged:
                merged[name] = clamp01(value)

    return merged


def _duration_seconds(audio: np.ndarray, sr: int) -> float:
    if sr <= 0:
        return 0.0
    return len(audio) / float(sr)


def _resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    audio = ensure_stereo(audio)

    if source_sr == target_sr:
        return audio

    channels = [
        librosa.resample(audio[:, channel], orig_sr=source_sr, target_sr=target_sr)
        for channel in range(audio.shape[1])
    ]
    sample_count = min(len(channel) for channel in channels)
    return np.column_stack([channel[:sample_count] for channel in channels]).astype(
        np.float32,
        copy=False,
    )


def _role_from_layer(layer: dict) -> str:
    role = layer.get("selected_role") or layer.get("role")

    if role:
        return str(role).lower()

    roles = layer.get("roles")
    if roles:
        first_role = roles[0]
        if isinstance(first_role, (list, tuple)) and first_role:
            return str(first_role[0]).lower()
        return str(first_role).lower()

    return "texture"


def _sample_from_layer(layer: dict) -> dict:
    sample = layer.get("sample")
    if isinstance(sample, dict):
        return sample
    return layer


def _path_from_layer(layer: dict, index_dir: Path) -> Path:
    path = layer.get("path")
    if path:
        return Path(path)

    sample = _sample_from_layer(layer)
    return resolve_sample_path(sample, index_dir)


def _filename_from_layer(layer: dict, index_dir: Path) -> str:
    sample = _sample_from_layer(layer)
    fallback = sample.get("id", "unknown.wav")
    return Path(sample.get("path", fallback)).name or _path_from_layer(layer, index_dir).name


def _gain_for_role(role: str, rng: np.random.Generator) -> float:
    low, high = ROLE_GAIN_RANGES.get(role, (-12.0, -6.0))
    return float(rng.uniform(low, high))


def _clip_window(
    audio: np.ndarray,
    sr: int,
    duration: float,
    role: str,
    rng: np.random.Generator,
) -> tuple[float, float, bool]:
    source_duration = _duration_seconds(audio, sr)
    duration = float(duration)
    role = role.lower()

    if role == "bed":
        return 0.0, duration, True

    if role == "low":
        max_start = min(1.0, max(0.0, duration * 0.1))
        return float(rng.uniform(0.0, max_start)) if max_start > 0 else 0.0, duration, True

    if role in {"texture", "motion"}:
        max_start = max(0.0, duration * 0.65)
        start = float(rng.uniform(0.0, max_start)) if max_start > 0 else 0.0
        remaining = max(0.0, duration - start)
        min_clip = min(remaining, max(1.0, duration * 0.25))
        max_clip = max(min_clip, remaining)
        clip_duration = float(rng.uniform(min_clip, max_clip)) if max_clip > min_clip else max_clip
        return start, clip_duration, True

    if role in {"foreground", "detail"}:
        max_accent = min(4.0, max(0.25, duration * 0.35))
        clip_duration = min(max_accent, source_duration if source_duration > 0 else max_accent)
        clip_duration = min(clip_duration, duration)
        max_start = max(0.0, duration - clip_duration)
        start = float(rng.uniform(0.0, max_start)) if max_start > 0 else 0.0
        return start, clip_duration, False

    max_start = max(0.0, duration * 0.5)
    start = float(rng.uniform(0.0, max_start)) if max_start > 0 else 0.0
    remaining = max(0.0, duration - start)
    clip_duration = min(remaining, max(1.0, source_duration or remaining))
    return start, clip_duration, True


def place_in_timeline(
    audio: np.ndarray,
    sr: int,
    duration: float,
    role: str,
    rng: np.random.Generator,
    gain_db: float,
) -> tuple[np.ndarray, dict]:
    if duration <= 0:
        raise ValueError("duration must be greater than 0.")

    audio = ensure_stereo(audio)
    target_samples = int(round(duration * sr))
    timeline = np.zeros((target_samples, 2), dtype=np.float32)

    start, clip_duration, should_loop = _clip_window(audio, sr, duration, role, rng)
    start = max(0.0, min(float(start), duration))
    clip_duration = max(0.0, min(float(clip_duration), duration - start))

    if should_loop:
        clip = trim_or_loop(audio, sr, clip_duration)
    else:
        clip_samples = int(round(clip_duration * sr))
        clip = audio[:clip_samples].astype(np.float32, copy=True)

    clip = apply_fade(clip, sr, fade_in=0.02, fade_out=0.08)
    clip = apply_gain(clip, gain_db)

    start_sample = int(round(start * sr))
    end_sample = min(target_samples, start_sample + len(clip))
    used_samples = max(0, end_sample - start_sample)

    if used_samples > 0:
        timeline[start_sample:end_sample] += clip[:used_samples]

    actual_duration = used_samples / float(sr) if sr > 0 else 0.0
    placement = {
        "role": role,
        "start": round(start_sample / float(sr), 4) if sr > 0 else 0.0,
        "duration": round(actual_duration, 4),
        "gain_db": round(float(gain_db), 2),
    }

    return timeline, placement


def sum_layers(layers: list[np.ndarray]) -> np.ndarray:
    if not layers:
        return np.zeros((0, 2), dtype=np.float32)

    max_samples = max(len(layer) for layer in layers)
    mix = np.zeros((max_samples, 2), dtype=np.float32)

    for layer in layers:
        layer = ensure_stereo(layer)
        mix[: len(layer)] += layer

    return mix


def master_limiter(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = ensure_stereo(audio)

    if audio.size == 0:
        return audio

    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = normalize_peak(audio, peak=1.0)

    drive = 1.15
    limited = np.tanh(audio * drive) / np.tanh(drive)
    return limited.astype(np.float32, copy=False)


def create_mix(
    prompt: str,
    index_dir: Path,
    output_path: Path,
    duration: float = 20.0,
    controls: dict[str, float] | None = None,
    seed: int = 42,
    pool_k: int = 20,
    normalize: bool = True,
    recipe_path: Path | None = None,
) -> dict:
    if duration <= 0:
        raise ValueError("duration must be greater than 0.")

    if pool_k < 1:
        raise ValueError("pool_k must be at least 1.")

    index_dir = Path(index_dir)
    output_path = Path(output_path)
    recipe_path = Path(recipe_path) if recipe_path is not None else output_path.with_suffix(
        ".recipe.json"
    )

    controls = _merge_controls(controls)
    layer_count = density_to_layer_count(controls["density"])
    selected = select_layers(prompt, index_dir, pool_k=pool_k, layers=layer_count)

    if not selected:
        raise ValueError("No valid layers were selected for this prompt.")

    rng = np.random.default_rng(seed)
    rendered_layers = []
    recipe_layers = []
    clips_used = []
    mix_sr = None

    for layer in selected:
        path = _path_from_layer(layer, index_dir)
        if not path.exists():
            raise FileNotFoundError(f"Selected audio file is missing: {path}")

        audio, sr = load_audio(path)
        if mix_sr is None:
            mix_sr = sr
        elif sr != mix_sr:
            audio = _resample_audio(audio, sr, mix_sr)
            sr = mix_sr

        transformed = transform_layer(audio, sr, controls)
        role = _role_from_layer(layer)
        gain_db = _gain_for_role(role, rng)
        timeline_layer, placement = place_in_timeline(
            transformed,
            sr,
            duration=duration,
            role=role,
            rng=rng,
            gain_db=gain_db,
        )

        filename = _filename_from_layer(layer, index_dir)
        clips_used.append(filename)
        rendered_layers.append(timeline_layer)
        recipe_layers.append(
            {
                "filename": filename,
                "path": str(path),
                "role": placement["role"],
                "score": float(layer.get("score", 0.0)),
                "start": placement["start"],
                "duration": placement["duration"],
                "gain_db": placement["gain_db"],
                "reason": layer.get("reason"),
            }
        )

    if mix_sr is None:
        raise ValueError("No audio could be loaded for the selected layers.")

    mix = sum_layers(rendered_layers)
    mix = master_limiter(mix, mix_sr)

    if normalize:
        mix = normalize_peak(mix)

    save_audio(output_path, mix, mix_sr)

    recipe = {
        "prompt": prompt,
        "controls": controls,
        "duration": float(duration),
        "seed": int(seed),
        "pool_k": int(pool_k),
        "layer_count": int(layer_count),
        "rendered_layer_count": len(recipe_layers),
        "sample_rate": int(mix_sr),
        "normalize": bool(normalize),
        "clips_used": clips_used,
        "layers": recipe_layers,
        "output": str(output_path),
    }

    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    with recipe_path.open("w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2)
        f.write("\n")

    recipe["recipe_path"] = str(recipe_path)
    return recipe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a complete TextureMap soundscape from a prompt."
    )
    parser.add_argument("prompt", help="Text prompt to render.")
    parser.add_argument("index_dir", type=Path, help="Directory containing index files.")
    parser.add_argument("output_path", type=Path, help="Output WAV path.")
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Output duration in seconds. Defaults to 20.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic placement.")
    parser.add_argument(
        "--pool-k",
        type=int,
        default=20,
        help="Number of retrieved candidates to analyze. Defaults to 20.",
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        default=None,
        help="Recipe JSON path. Defaults to output_path with .recipe.json suffix.",
    )
    parser.add_argument("--brightness", type=float, default=0.5)
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--distance", type=float, default=0.5)
    parser.add_argument("--movement", type=float, default=0.5)
    parser.add_argument("--tension", type=float, default=0.5)
    parser.add_argument("--texture", type=float, default=0.5)

    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        default=True,
        help="Peak-normalize the master before saving. Enabled by default.",
    )
    normalize_group.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Skip final peak normalization.",
    )

    args = parser.parse_args()

    controls = {
        "brightness": args.brightness,
        "density": args.density,
        "distance": args.distance,
        "movement": args.movement,
        "tension": args.tension,
        "texture": args.texture,
    }

    try:
        recipe = create_mix(
            prompt=args.prompt,
            index_dir=args.index_dir,
            output_path=args.output_path,
            duration=args.duration,
            controls=controls,
            seed=args.seed,
            pool_k=args.pool_k,
            normalize=args.normalize,
            recipe_path=args.recipe,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print("Rendered TextureMap mix.")
    print(f"Prompt: {args.prompt}")
    print(f"Output: {args.output_path}")
    print(f"Recipe: {recipe['recipe_path']}")
    print(f"Duration: {recipe['duration']:.2f} seconds")
    print(f"Sample rate: {recipe['sample_rate']} Hz")
    print(f"Density selected {recipe['layer_count']} target layers.")
    print(f"Rendered layers: {recipe['rendered_layer_count']}")
    print("Controls:")
    for name in CONTROL_NAMES:
        print(f"  {name}: {recipe['controls'][name]:.2f}")


if __name__ == "__main__":
    main()
