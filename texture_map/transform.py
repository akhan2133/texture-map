import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pedalboard import (
    Delay,
    Distortion,
    Gain,
    HighpassFilter,
    HighShelfFilter,
    Limiter,
    LowpassFilter,
    Pedalboard,
    PitchShift,
    Reverb,
)


EPSILON = 1e-8


def clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _as_float32(audio: np.ndarray) -> np.ndarray:
    return np.asarray(audio, dtype=np.float32)


def _to_pedalboard(audio: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(ensure_stereo(audio).T, dtype=np.float32)


def _from_pedalboard(audio: np.ndarray) -> np.ndarray:
    return ensure_stereo(np.asarray(audio).T)


def _run_board(audio: np.ndarray, sr: int, effects: list) -> np.ndarray:
    if audio.size == 0 or not effects:
        return ensure_stereo(audio)

    processed = Pedalboard(effects)(_to_pedalboard(audio), sr)
    return _from_pedalboard(processed)


def _blend(dry: np.ndarray, wet: np.ndarray, amount: float) -> np.ndarray:
    amount = clamp01(amount)
    sample_count = min(len(dry), len(wet))

    if sample_count == 0:
        return ensure_stereo(dry)

    dry = ensure_stereo(dry)[:sample_count]
    wet = ensure_stereo(wet)[:sample_count]
    return _as_float32((dry * (1.0 - amount)) + (wet * amount))


def _stereo_width(audio: np.ndarray, width: float) -> np.ndarray:
    audio = ensure_stereo(audio)
    width = float(np.clip(width, 0.0, 2.0))

    mid = np.mean(audio, axis=1, keepdims=True)
    side = (audio[:, :1] - audio[:, 1:2]) * 0.5

    left = mid + side * width
    right = mid - side * width
    return _as_float32(np.hstack([left, right]))


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return ensure_stereo(audio), int(sr)


def save_audio(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = ensure_stereo(audio)
    current_peak = float(np.max(np.abs(audio))) if audio.size else 0.0

    if current_peak > 1.0:
        audio = normalize_peak(audio)
    else:
        audio = np.clip(audio, -1.0, 1.0)

    sf.write(path, _as_float32(audio), sr)


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    audio = _as_float32(audio)

    if audio.ndim == 1:
        return np.column_stack([audio, audio]).astype(np.float32, copy=False)

    if audio.ndim != 2:
        raise ValueError("audio must be a 1D mono or 2D channel/sample array")

    if audio.shape[1] == 2:
        return audio.astype(np.float32, copy=False)

    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1).astype(np.float32, copy=False)

    if audio.shape[0] in (1, 2) and audio.shape[1] > 2:
        return ensure_stereo(audio.T)

    mono = np.mean(audio, axis=1)
    return np.column_stack([mono, mono]).astype(np.float32, copy=False)


def normalize_peak(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    audio = ensure_stereo(audio)
    peak = float(max(0.0, peak))
    current_peak = float(np.max(np.abs(audio))) if audio.size else 0.0

    if current_peak <= EPSILON or peak <= EPSILON:
        return audio.astype(np.float32, copy=True)

    return _as_float32(audio * (peak / current_peak))


def apply_gain(audio: np.ndarray, db: float) -> np.ndarray:
    audio = ensure_stereo(audio)
    gain = 10.0 ** (float(db) / 20.0)
    return _as_float32(audio * gain)


def apply_pan(audio: np.ndarray, pan: float) -> np.ndarray:
    audio = ensure_stereo(audio)
    pan = float(np.clip(pan, -1.0, 1.0))

    angle = (pan + 1.0) * (np.pi / 4.0)
    left_gain = np.cos(angle)
    right_gain = np.sin(angle)

    return _as_float32(audio * np.array([left_gain, right_gain], dtype=np.float32))


def apply_fade(
    audio: np.ndarray,
    sr: int,
    fade_in: float = 0.01,
    fade_out: float = 0.05,
) -> np.ndarray:
    audio = ensure_stereo(audio).copy()
    sample_count = len(audio)

    if sample_count == 0:
        return audio

    fade_in_samples = min(int(max(0.0, fade_in) * sr), sample_count)
    fade_out_samples = min(int(max(0.0, fade_out) * sr), sample_count)

    if fade_in_samples > 0:
        audio[:fade_in_samples] *= np.linspace(0.0, 1.0, fade_in_samples)[:, None]

    if fade_out_samples > 0:
        audio[-fade_out_samples:] *= np.linspace(1.0, 0.0, fade_out_samples)[:, None]

    return _as_float32(audio)


def trim_or_loop(audio: np.ndarray, sr: int, duration: float) -> np.ndarray:
    audio = ensure_stereo(audio)
    target_samples = int(max(0.0, duration) * sr)

    if target_samples <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    if len(audio) == 0:
        return np.zeros((target_samples, 2), dtype=np.float32)

    if len(audio) >= target_samples:
        return audio[:target_samples].astype(np.float32, copy=True)

    repeats = int(np.ceil(target_samples / len(audio)))
    looped = np.tile(audio, (repeats, 1))[:target_samples]
    return _as_float32(looped)


def low_pass(audio: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    cutoff_hz = float(np.clip(cutoff_hz, 20.0, sr * 0.45))
    return _run_board(audio, sr, [LowpassFilter(cutoff_frequency_hz=cutoff_hz)])


def high_pass(audio: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    cutoff_hz = float(np.clip(cutoff_hz, 20.0, sr * 0.45))
    return _run_board(audio, sr, [HighpassFilter(cutoff_frequency_hz=cutoff_hz)])


def apply_reverb(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    amount = clamp01(amount)

    if amount <= EPSILON:
        return ensure_stereo(audio)

    return _run_board(
        audio,
        sr,
        [
            Reverb(
                room_size=0.25 + amount * 0.55,
                damping=0.45 + amount * 0.35,
                wet_level=amount * 0.35,
                dry_level=1.0 - amount * 0.12,
                width=0.6 + amount * 0.4,
            )
        ],
    )


def apply_delay(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    amount = clamp01(amount)

    if amount <= EPSILON:
        return ensure_stereo(audio)

    return _run_board(
        audio,
        sr,
        [
            Delay(
                delay_seconds=0.08 + amount * 0.28,
                feedback=amount * 0.35,
                mix=amount * 0.35,
            )
        ],
    )


def reverse_audio(audio: np.ndarray) -> np.ndarray:
    return ensure_stereo(audio)[::-1].astype(np.float32, copy=True)


def apply_time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    audio = ensure_stereo(audio)
    rate = float(max(rate, EPSILON))

    if len(audio) == 0 or abs(rate - 1.0) <= EPSILON:
        return audio.astype(np.float32, copy=True)

    channels = [
        librosa.effects.time_stretch(audio[:, channel], rate=rate)
        for channel in range(audio.shape[1])
    ]
    sample_count = min(len(channel) for channel in channels)
    stretched = np.column_stack([channel[:sample_count] for channel in channels])
    return _as_float32(stretched)


def apply_pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    if abs(semitones) <= EPSILON:
        return ensure_stereo(audio)

    return _run_board(audio, sr, [PitchShift(semitones=float(semitones))])


def apply_autopan(
    audio: np.ndarray,
    sr: int,
    amount: float,
    rate_hz: float = 0.1,
) -> np.ndarray:
    audio = ensure_stereo(audio)
    amount = clamp01(amount)

    if amount <= EPSILON or len(audio) == 0:
        return audio.astype(np.float32, copy=True)

    seconds = np.arange(len(audio), dtype=np.float32) / float(sr)
    lfo = np.sin(2.0 * np.pi * float(rate_hz) * seconds)
    pan = lfo * amount * 0.85

    angle = (pan + 1.0) * (np.pi / 4.0)
    gains = np.column_stack([np.cos(angle), np.sin(angle)]).astype(np.float32)
    return _as_float32(audio * gains)


def apply_brightness(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Move the layer from dark and tucked back to clear and airy."""
    amount = clamp01(amount)

    if amount < 0.49:
        cutoff = np.interp(amount, [0.0, 0.5], [800.0, 12000.0])
        return low_pass(audio, sr, cutoff)

    if amount <= 0.51:
        return ensure_stereo(audio)

    lift = (amount - 0.5) * 2.0
    bright = _run_board(
        audio,
        sr,
        [
            HighpassFilter(cutoff_frequency_hz=35.0 + lift * 120.0),
            HighShelfFilter(cutoff_frequency_hz=3500.0, gain_db=lift * 5.0),
            Gain(gain_db=lift * 1.0),
        ],
    )
    return _blend(audio, bright, 0.75)


def apply_distance(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Push the layer farther away by making it quieter, darker, narrower, and wetter."""
    amount = clamp01(amount)

    if amount <= EPSILON:
        return ensure_stereo(audio)

    output = apply_gain(audio, -amount * 9.0)
    output = low_pass(output, sr, np.interp(amount, [0.0, 1.0], [18000.0, 2500.0]))
    output = _stereo_width(output, 1.0 - amount * 0.65)
    output = apply_reverb(output, sr, amount * 0.8)
    return _as_float32(output)


def apply_movement(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Add slow stereo motion, from almost still to an obvious but gentle autopan."""
    amount = clamp01(amount)
    return apply_autopan(audio, sr, amount=amount, rate_hz=0.04 + amount * 0.16)


def apply_tension(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Make the layer sharper and more urgent with light filtering and saturation."""
    amount = clamp01(amount)

    if amount <= EPSILON:
        return ensure_stereo(audio)

    filtered = _run_board(
        audio,
        sr,
        [
            HighpassFilter(cutoff_frequency_hz=40.0 + amount * 260.0),
            HighShelfFilter(cutoff_frequency_hz=2200.0, gain_db=amount * 4.0),
        ],
    )
    driven = _run_board(filtered, sr, [Distortion(drive_db=amount * 8.0)])
    output = _blend(filtered, driven, amount * 0.28)
    return _run_board(output, sr, [Limiter(threshold_db=-0.2)])


def apply_texture(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Smear and roughen the layer with delay, plus a reversed shadow at higher settings."""
    amount = clamp01(amount)

    if amount <= EPSILON:
        return ensure_stereo(audio)

    output = apply_delay(audio, sr, amount)

    if amount > 0.55:
        reverse_amount = (amount - 0.55) / 0.45
        reversed_shadow = apply_fade(reverse_audio(output), sr, fade_in=0.02, fade_out=0.12)
        output = _blend(output, reversed_shadow, reverse_amount * 0.25)

    return _run_board(output, sr, [Limiter(threshold_db=-0.2)])


def transform_layer(audio: np.ndarray, sr: int, controls: dict[str, float]) -> np.ndarray:
    output = ensure_stereo(audio)

    output = apply_brightness(output, sr, controls.get("brightness", 0.5))
    output = apply_tension(output, sr, controls.get("tension", 0.0))
    output = apply_texture(output, sr, controls.get("texture", 0.0))
    output = apply_movement(output, sr, controls.get("movement", 0.0))
    output = apply_distance(output, sr, controls.get("distance", 0.0))

    return _run_board(output, sr, [Limiter(threshold_db=-0.2)])


def _duration_seconds(audio: np.ndarray, sr: int) -> float:
    if sr <= 0:
        return 0.0
    return len(audio) / float(sr)


def main() -> None:
    parser = argparse.ArgumentParser (
        description="Transform one TextureMap audio layer with simple producer controls."
    )
    parser.add_argument("input_path", type=Path, help="Input WAV path.")
    parser.add_argument("output_path", type=Path, help="Output WAV path.")
    parser.add_argument("--brightness", type=float, default=0.5)
    parser.add_argument("--distance", type=float, default=0.0)
    parser.add_argument("--movement", type=float, default=0.0)
    parser.add_argument("--tension", type=float, default=0.0)
    parser.add_argument("--texture", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Peak-normalize the transformed audio before saving.",
    )

    args = parser.parse_args()

    controls = {
        "brightness": clamp01(args.brightness),
        "distance": clamp01(args.distance),
        "movement": clamp01(args.movement),
        "tension": clamp01(args.tension),
        "texture": clamp01(args.texture),
    }

    audio, sr = load_audio(args.input_path)

    if args.duration is not None:
        if args.duration <= 0:
            parser.error("--duration must be greater than 0.")
        audio = trim_or_loop(audio, sr, args.duration)

    transformed = transform_layer(audio, sr, controls)

    if args.normalize:
        transformed = normalize_peak(transformed)

    save_audio(args.output_path, transformed, sr)

    print("Transformed one TextureMap layer.")
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {_duration_seconds(transformed, sr):.2f} seconds")
    print("Controls:")
    for name, value in controls.items():
        print(f"  {name}: {value:.2f}")
    if args.duration is not None:
        print(f"Target duration: {args.duration:.2f} seconds")
    print(f"Normalize before save: {args.normalize}")


if __name__ == "__main__":
    main()
