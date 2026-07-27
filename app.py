import hashlib
import io
import re
from pathlib import Path
from typing import Any

import soundfile as sf
import streamlit as st

from texture_map.mix import create_mix
from texture_map.select_layers import resolve_sample_path, select_layers
from texture_map.waveform import WaveSurferOptions, region_wavesurfer


BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "index"
OUTPUTS_DIR = BASE_DIR / "outputs"
INDEX_FILES = (INDEX_DIR / "embeddings.npy", INDEX_DIR / "metadata.json")
LAYER_WIDGET_PREFIXES = (
    "expand_button_",
    "expanded_",
    "include_",
    "region_wave_",
)


def sanitize_filename(prompt: str, max_length: int = 70) -> str:
    """Turn a prompt into a short, filesystem-safe filename stem."""
    stem = prompt.strip().lower()
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^a-z0-9_-]", "", stem)
    stem = re.sub(r"_+", "_", stem).strip("_-")
    stem = stem[:max_length].rstrip("_-")
    return stem or "texture"


def validate_index() -> list[Path]:
    return [path for path in INDEX_FILES if not path.is_file()]


def available_output_path(prompt: str, seed: int) -> Path:
    base_stem = f"{sanitize_filename(prompt)}_seed{seed}"
    candidate = OUTPUTS_DIR / f"{base_stem}.wav"
    suffix = 2

    while candidate.exists() or candidate.with_suffix(".recipe.json").exists():
        candidate = OUTPUTS_DIR / f"{base_stem}_{suffix}.wav"
        suffix += 1

    return candidate


def _sample_from_layer(layer: dict[str, Any]) -> dict[str, Any]:
    sample = layer.get("sample")
    return sample if isinstance(sample, dict) else layer


def layer_path(layer: dict[str, Any]) -> Path:
    explicit_path = layer.get("path")
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_absolute() and not path.exists():
            project_path = BASE_DIR / path
            if project_path.exists():
                path = project_path
        return path
    return resolve_sample_path(_sample_from_layer(layer), INDEX_DIR)


def layer_filename(layer: dict[str, Any]) -> str:
    if layer.get("filename"):
        return str(layer["filename"])
    sample = _sample_from_layer(layer)
    return Path(sample.get("path", sample.get("id", "unknown.wav"))).name


def layer_role(layer: dict[str, Any]) -> str:
    role = layer.get("selected_role") or layer.get("role")
    if role:
        return str(role)
    roles = layer.get("roles")
    if isinstance(roles, list) and roles:
        first = roles[0]
        if isinstance(first, (list, tuple)) and first:
            return str(first[0])
        return str(first)
    return "texture"


def source_duration(layer: dict[str, Any], path: Path) -> float:
    features = layer.get("features")
    if isinstance(features, dict) and features.get("duration") is not None:
        duration = float(features["duration"])
    else:
        sample = _sample_from_layer(layer)
        duration_value = sample.get("duration")
        duration = float(duration_value) if duration_value is not None else 0.0

    if duration <= 0:
        try:
            info = sf.info(path)
        except Exception as exc:
            raise ValueError(f"Could not read source duration for {path.name}: {exc}") from exc
        duration = float(info.duration)

    if duration <= 0:
        raise ValueError(f"Source duration is unknown for {path.name}.")
    return duration


def stable_layer_id(layer: dict[str, Any], selection_index: int) -> str:
    path = layer_path(layer)
    stable_path = str(path.resolve(strict=False))
    digest = hashlib.sha256(f"{selection_index}:{stable_path}".encode("utf-8")).hexdigest()
    return digest[:14]


def prepare_proposed_layer(layer: dict[str, Any], selection_index: int) -> dict[str, Any]:
    prepared = dict(layer)
    path = layer_path(prepared)
    if not path.is_file():
        raise FileNotFoundError(f"Selected audio file is missing: {path}")
    duration = source_duration(prepared, path)
    prepared["path"] = str(path)
    prepared["filename"] = layer_filename(prepared)
    prepared["role"] = layer_role(prepared)
    prepared["source_duration"] = duration
    prepared["layer_id"] = stable_layer_id(prepared, selection_index)
    return prepared


def initialize_session_state() -> None:
    defaults = {
        "layer_regions": {},
        "latest_output_path": None,
        "latest_recipe": None,
        "latest_recipe_path": None,
        "proposed_layers": [],
        "selection_pool_size": None,
        "selection_prompt": None,
        "selection_target_count": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_palette_widget_state(layers: list[dict[str, Any]]) -> None:
    old_ids = {
        str(layer.get("layer_id"))
        for layer in layers
        if isinstance(layer, dict) and layer.get("layer_id")
    }
    for layer_id in old_ids:
        for prefix in LAYER_WIDGET_PREFIXES:
            st.session_state.pop(f"{prefix}{layer_id}", None)


def store_proposed_palette(
    layers: list[dict[str, Any]],
    prompt: str,
    target_count: int,
    pool_size: int,
) -> None:
    clear_palette_widget_state(st.session_state.proposed_layers)
    st.session_state.proposed_layers = layers
    st.session_state.selection_prompt = prompt
    st.session_state.selection_target_count = int(target_count)
    st.session_state.selection_pool_size = int(pool_size)
    st.session_state.layer_regions = {}

    for layer in layers:
        layer_id = layer["layer_id"]
        duration = float(layer["source_duration"])
        st.session_state[f"include_{layer_id}"] = True
        st.session_state[f"expanded_{layer_id}"] = False
        st.session_state.layer_regions[layer_id] = {
            "source_start": 0.0,
            "source_end": duration,
        }


def build_layer_rows(layers: Any) -> list[dict[str, Any]]:
    if not isinstance(layers, list):
        return []

    rows = []
    for layer in layers:
        if isinstance(layer, dict):
            rows.append(
                {
                    "Filename": layer.get("filename"),
                    "Role": layer.get("role"),
                    "Similarity score": layer.get("score"),
                    "Source start (s)": layer.get("source_start"),
                    "Source end (s)": layer.get("source_end"),
                    "Selected source duration (s)": layer.get("source_region_duration"),
                    "Timeline start (s)": layer.get("start"),
                    "Rendered duration (s)": layer.get("duration"),
                    "Gain (dB)": layer.get("gain_db"),
                }
            )
        else:
            rows.append({"Filename": str(layer)})
    return rows


def clip_label(clip: Any) -> str:
    if isinstance(clip, dict):
        return str(clip.get("filename") or clip.get("path") or clip.get("id") or clip)
    return str(clip)


def metric_value(value: Any, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value}{suffix}"


def render_results(recipe: dict[str, Any], output_path: Path, recipe_path: Path) -> None:
    st.divider()
    st.subheader("Latest Texture")
    st.success(f"Rendered to `{output_path}`")

    metrics = st.columns(7)
    metrics[0].metric("Duration", metric_value(recipe.get("duration"), " s"))
    metrics[1].metric("Sample rate", metric_value(recipe.get("sample_rate"), " Hz"))
    metrics[2].metric(
        "Requested layers",
        metric_value(recipe.get("requested_layer_count", recipe.get("layer_count"))),
    )
    metrics[3].metric("Curated layers", metric_value(recipe.get("curated_layer_count")))
    metrics[4].metric("Rendered layers", metric_value(recipe.get("rendered_layer_count")))
    metrics[5].metric("Seed", metric_value(recipe.get("seed")))
    metrics[6].metric("Pool size", metric_value(recipe.get("pool_k")))

    if output_path.is_file():
        st.audio(str(output_path), format="audio/wav")
        try:
            wav_bytes = output_path.read_bytes()
            st.download_button(
                label="Download WAV",
                data=wav_bytes,
                file_name=output_path.name,
                mime="audio/wav",
            )
        except OSError as exc:
            st.warning(f"The WAV could not be read for download: {exc}")
    else:
        st.error(f"The rendered WAV is missing: {output_path}")

    if recipe_path.is_file():
        try:
            recipe_bytes = recipe_path.read_bytes()
            st.download_button(
                label="Download Recipe JSON",
                data=recipe_bytes,
                file_name=recipe_path.name,
                mime="application/json",
            )
        except OSError as exc:
            st.warning(f"The recipe JSON could not be read for download: {exc}")
    else:
        st.warning(f"The recipe JSON is unavailable: {recipe_path}")

    clips = recipe.get("clips_used")
    if isinstance(clips, list) and clips:
        st.subheader("Selected Clips")
        for clip in clips:
            st.write(f"- {clip_label(clip)}")
    else:
        st.info("No selected clip information is available in this recipe.")

    layer_rows = build_layer_rows(recipe.get("layers"))
    if layer_rows:
        with st.expander("Layer Details", expanded=True):
            st.dataframe(layer_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No rendered layer details are available in this recipe.")

    with st.expander("Full Recipe JSON"):
        st.json(recipe)


def _region_from_component_state(
    state: Any,
    duration: float,
) -> tuple[float, float] | None:
    if not isinstance(state, dict):
        return None
    regions = state.get("regions")
    if not isinstance(regions, list) or not regions or not isinstance(regions[0], dict):
        return None
    try:
        start = float(regions[0]["start"])
        end = float(regions[0]["end"])
    except (KeyError, TypeError, ValueError):
        return None
    start = max(0.0, min(start, duration))
    end = max(0.0, min(end, duration))
    if end <= start:
        return None
    return round(start, 3), round(end, 3)


def _region_bounds_changed(
    current: tuple[float, float],
    updated: tuple[float, float],
    tolerance: float = 0.0005,
) -> bool:
    return any(abs(old - new) > tolerance for old, new in zip(current, updated))


def synchronize_waveform_region_state(layer_id: str, duration: float) -> None:
    """Apply a completed waveform edit before rendering the compact preview."""
    updated = _region_from_component_state(
        st.session_state.get(f"region_wave_{layer_id}"),
        duration,
    )
    if updated is None:
        return

    stored = st.session_state.layer_regions.get(
        layer_id,
        {"source_start": 0.0, "source_end": duration},
    )
    current = (float(stored["source_start"]), float(stored["source_end"]))
    if _region_bounds_changed(current, updated):
        st.session_state.layer_regions[layer_id] = {
            "source_start": updated[0],
            "source_end": updated[1],
        }


def waveform_region_selector(
    layer_id: str,
    audio_path: Path,
    duration: float,
    initial_start: float,
    initial_end: float,
) -> tuple[float, float]:
    """Render a draggable region and return its bounds after editing finishes."""
    region = {
        "id": f"selection_{layer_id}",
        "start": float(initial_start),
        "end": float(initial_end),
        "content": "Selected source",
        "color": "rgba(61, 157, 242, 0.28)",
        "drag": True,
        "resize": True,
        "resizeStart": True,
        "resizeEnd": True,
    }
    try:
        state = region_wavesurfer(
            audio_src=str(audio_path),
            regions=[region],
            key=f"region_wave_{layer_id}",
            wave_options=WaveSurferOptions(
                waveColor="#8d99ae",
                progressColor="#8d99ae",
                cursorWidth=0,
                height=150,
                minPxPerSec=1,
                fillParent=True,
                autoScroll=False,
                autoCenter=False,
                hideScrollbar=False,
                regionOpacity=0.28,
                interact=False,
                dragToSeek=False,
            ),
            region_colormap="cool",
            show_controls=False,
            plugins=["regions", "timeline"],
        )
    except Exception as exc:
        st.warning(f"The waveform editor failed for this layer: {exc}")
        return initial_start, initial_end

    updated = _region_from_component_state(state, duration)
    return updated if updated is not None else (initial_start, initial_end)


def compact_source_player(
    audio_path: Path,
    source_start: float,
    source_end: float,
) -> None:
    """Preview only the current selected source region in native Streamlit audio."""
    try:
        modified_ns = audio_path.stat().st_mtime_ns
        preview = selected_region_preview(
            str(audio_path),
            modified_ns,
            round(float(source_start), 3),
            round(float(source_end), 3),
        )
        st.audio(preview, format="audio/wav")
    except Exception as exc:
        st.warning(
            "The selected source region could not be previewed: "
            f"{exc}"
        )


@st.cache_data(show_spinner=False)
def selected_region_preview(
    audio_path: str,
    modified_ns: int,
    source_start: float,
    source_end: float,
) -> bytes:
    """Create an in-memory WAV; modified_ns participates in the cache key."""
    _ = modified_ns
    path = Path(audio_path)
    with sf.SoundFile(path) as source:
        sample_rate = int(source.samplerate)
        start_frame = max(0, min(int(round(source_start * sample_rate)), len(source)))
        end_frame = max(0, min(int(round(source_end * sample_rate)), len(source)))
        if end_frame <= start_frame:
            raise ValueError("The selected preview region is empty.")
        source.seek(start_frame)
        audio = source.read(end_frame - start_frame, dtype="float32", always_2d=True)

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def curated_layer_count(layers: list[dict[str, Any]]) -> int:
    return sum(
        bool(st.session_state.get(f"include_{layer['layer_id']}", True))
        for layer in layers
    )


def render_proposed_layer(layer: dict[str, Any], layer_number: int) -> None:
    layer_id = layer["layer_id"]
    path = layer_path(layer)
    filename = layer_filename(layer)
    role = layer_role(layer)
    score = float(layer.get("score", 0.0))
    duration = float(layer["source_duration"])
    include_key = f"include_{layer_id}"
    expanded_key = f"expanded_{layer_id}"
    synchronize_waveform_region_state(layer_id, duration)
    region = st.session_state.layer_regions[layer_id]
    source_start = float(region["source_start"])
    source_end = float(region["source_end"])

    with st.container(border=True):
        transport_col, info_col, include_col, expand_col = st.columns(
            [3.2, 6.0, 1.5, 1.8],
            vertical_alignment="center",
        )
        with transport_col:
            compact_source_player(path, source_start, source_end)
        with info_col:
            st.write(f"**{layer_number}. {filename}**")
            st.caption(f"{role} · {score:.3f} · {duration:.2f} s")
        with include_col:
            st.checkbox("Include", key=include_key)
        with expand_col:
            expanded = bool(st.session_state.get(expanded_key, False))
            label = "Collapse ▴" if expanded else "Expand ▾"
            if st.button(label, key=f"expand_button_{layer_id}", use_container_width=True):
                st.session_state[expanded_key] = not expanded
                st.rerun()

        if st.session_state.get(expanded_key, False):
            st.divider()
            details = st.columns(4)
            details[0].metric("Role", role)
            details[1].metric("Similarity", f"{score:.3f}")
            details[2].metric("Source duration", f"{duration:.2f} s")
            details[3].write("**Selection reason**")
            details[3].caption(str(layer.get("reason") or "No reason recorded."))

            start, end = waveform_region_selector(
                layer_id=layer_id,
                audio_path=path,
                duration=duration,
                initial_start=source_start,
                initial_end=source_end,
            )
            st.caption(
                "Drag the purple selection to move it; drag either edge to resize it."
            )
            updated = (start, end)
            if _region_bounds_changed((source_start, source_end), updated):
                st.session_state.layer_regions[layer_id] = {
                    "source_start": start,
                    "source_end": end,
                }
                st.rerun()

            selected_duration = end - start
            st.write(
                f"Selected region: **{start:.2f}s–{end:.2f}s** "
                f"({selected_duration:.2f}s)"
            )
            st.caption("The compact player above previews this selected region.")


def validate_curated_layers(
    layers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    curated = []
    for layer in layers:
        layer_id = layer["layer_id"]
        if not st.session_state.get(f"include_{layer_id}", True):
            continue

        path = layer_path(layer)
        if not path.is_file():
            raise FileNotFoundError(f"Included source audio is missing: {path}")

        duration = float(layer["source_duration"])
        region = st.session_state.layer_regions.get(layer_id, {})
        start = float(region.get("source_start", 0.0))
        end = float(region.get("source_end", duration))
        if start < 0:
            raise ValueError(f"{path.name}: source start cannot be negative.")
        if end > duration + 1e-3:
            raise ValueError(f"{path.name}: source end exceeds the source duration.")
        if end <= start:
            raise ValueError(f"{path.name}: source end must be after source start.")

        curated_layer = dict(layer)
        curated_layer["path"] = str(path)
        curated_layer["source_start"] = max(0.0, start)
        curated_layer["source_end"] = min(duration, end)
        curated.append(curated_layer)

    if not curated:
        raise ValueError("Include at least one proposed layer before generating a texture.")
    return curated


def render_discovery_controls() -> tuple[str, int, int, bool]:
    prompt = st.text_area(
        "Sound-design prompt",
        value="wet neon alley with far sirens",
        help="Describe the atmosphere, material, location, or motion you want to hear.",
    )
    controls = st.columns([2, 2, 1], vertical_alignment="bottom")
    with controls[0]:
        target_layers = st.slider(
            "Proposed layers",
            min_value=2,
            max_value=10,
            value=5,
            step=1,
            help="The number of layers TextureMap should initially propose.",
        )
    with controls[1]:
        pool_k = int(
            st.number_input(
                "Retrieval pool size",
                min_value=1,
                value=20,
                step=1,
                help="How many prompt matches Stage 4 analyzes before balancing the palette.",
            )
        )
    with controls[2]:
        find_clicked = st.button("Find Layers", type="primary", use_container_width=True)
    return prompt, int(target_layers), pool_k, find_clicked


def find_layers(prompt: str, target_layers: int, pool_k: int) -> None:
    clean_prompt = prompt.strip()
    if not clean_prompt:
        st.error("Enter a sound-design prompt before finding layers.")
        return
    if not 2 <= target_layers <= 10:
        st.error("The proposed layer count must be between 2 and 10.")
        return
    if pool_k < 1:
        st.error("The retrieval pool size must be at least 1.")
        return
    if pool_k < target_layers:
        st.error("The retrieval pool cannot be smaller than the proposed layer count.")
        return
    missing_index_files = validate_index()
    if missing_index_files:
        missing_names = ", ".join(path.name for path in missing_index_files)
        st.error(
            "TextureMap's sample index is missing. Run the preparation and embedding "
            f"stages before finding layers. Missing: {missing_names}"
        )
        return

    try:
        with st.spinner("Retrieving and selecting a balanced layer palette..."):
            selected = select_layers(
                clean_prompt,
                INDEX_DIR,
                pool_k=pool_k,
                layers=target_layers,
            )
            proposed = [
                prepare_proposed_layer(layer, index)
                for index, layer in enumerate(selected)
            ]
        if not proposed:
            raise ValueError("No valid layers remained after retrieval and feature analysis.")
        store_proposed_palette(proposed, clean_prompt, target_layers, pool_k)
        st.success(f"Found {len(proposed)} proposed layers.")
    except Exception as exc:
        st.error(f"Layer discovery failed: {exc}")


def render_producer_controls() -> tuple[dict[str, float], int, int]:
    st.subheader("Producer Controls")
    left, right = st.columns(2)
    with left:
        brightness = st.slider(
            "Brightness", 0.0, 1.0, 0.5, 0.05, help="Darker to brighter."
        )
        distance = st.slider(
            "Distance",
            0.0,
            1.0,
            0.5,
            0.05,
            help="Close and dry to distant and reverberant.",
        )
        tension = st.slider(
            "Tension",
            0.0,
            1.0,
            0.5,
            0.05,
            help="Smooth to harsher and more saturated.",
        )
    with right:
        movement = st.slider(
            "Movement",
            0.0,
            1.0,
            0.5,
            0.05,
            help="Static to moving stereo image.",
        )
        texture = st.slider(
            "Texture",
            0.0,
            1.0,
            0.5,
            0.05,
            help="Clean to smeared and reverse-blended.",
        )

    st.subheader("Render Settings")
    settings = st.columns(2)
    with settings[0]:
        duration = st.slider("Duration (seconds)", 5, 20, 20)
    with settings[1]:
        seed = int(st.number_input("Seed", value=42, step=1))

    target = int(st.session_state.selection_target_count)
    controls = {
        "brightness": brightness,
        "density": (target - 2) / 8.0,
        "distance": distance,
        "movement": movement,
        "tension": tension,
        "texture": texture,
    }
    return controls, int(duration), seed


def generate_texture(
    layers: list[dict[str, Any]],
    controls: dict[str, float],
    duration: int,
    seed: int,
) -> None:
    try:
        curated = validate_curated_layers(layers)
        clean_prompt = str(st.session_state.selection_prompt)
        pool_k = int(st.session_state.selection_pool_size)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = available_output_path(clean_prompt, seed)

        with st.spinner("Transforming and mixing the curated layer palette..."):
            recipe = create_mix(
                prompt=clean_prompt,
                index_dir=INDEX_DIR,
                output_path=output_path,
                duration=float(duration),
                controls=controls,
                seed=seed,
                pool_k=pool_k,
                normalize=True,
                selected_layers=curated,
            )

        if not output_path.is_file():
            raise FileNotFoundError(
                f"Rendering completed, but the output WAV was not found: {output_path}"
            )
        returned_recipe_path = recipe.get("recipe_path")
        recipe_path = (
            Path(returned_recipe_path)
            if returned_recipe_path
            else output_path.with_suffix(".recipe.json")
        )
        if not recipe_path.is_absolute():
            recipe_path = BASE_DIR / recipe_path
        if not recipe_path.is_file():
            raise FileNotFoundError(
                "Rendering completed, but the recipe JSON was not found: "
                f"{recipe_path}"
            )

        st.session_state.latest_recipe = recipe
        st.session_state.latest_output_path = str(output_path)
        st.session_state.latest_recipe_path = str(recipe_path)
    except Exception as exc:
        st.error(f"Texture rendering failed: {exc}")


def main() -> None:
    st.set_page_config(page_title="TextureMap", layout="wide")
    initialize_session_state()

    st.title("TextureMap")
    st.write(
        "TextureMap retrieves sounds from an indexed sample library, lets you curate "
        "their source regions, and transforms the chosen layers into a soundscape."
    )

    prompt, target_layers, pool_k, find_clicked = render_discovery_controls()
    if find_clicked:
        find_layers(prompt, target_layers, pool_k)

    layers = st.session_state.proposed_layers
    if layers:
        if (
            prompt.strip() != st.session_state.selection_prompt
            or target_layers != st.session_state.selection_target_count
            or pool_k != st.session_state.selection_pool_size
        ):
            st.caption(
                "Discovery settings have changed. Click **Find Layers** to replace the "
                "current proposed palette; it will not update automatically."
            )

        st.subheader("Proposed Layers")
        included = curated_layer_count(layers)
        st.write(
            f"**{included} of {len(layers)} layers included.** "
            "Use the compact source transports to browse, then expand only the layers "
            "whose source regions you want to refine."
        )
        for index, layer in enumerate(layers, start=1):
            render_proposed_layer(layer, index)

        controls, duration, seed = render_producer_controls()
        if st.button("Generate Texture", type="primary"):
            generate_texture(layers, controls, duration, seed)

    if (
        st.session_state.latest_recipe is not None
        and st.session_state.latest_output_path
        and st.session_state.latest_recipe_path
    ):
        render_results(
            st.session_state.latest_recipe,
            Path(st.session_state.latest_output_path),
            Path(st.session_state.latest_recipe_path),
        )


if __name__ == "__main__":
    main()
