"""TextureMap's WaveSurfer bridge with editable-region change events."""

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components
from streamlit_wavesurfer import WaveSurferOptions
from streamlit_wavesurfer.utils import (
    DEFAULT_PLUGINS,
    WaveSurferPluginConfigurationList,
    audio_to_base64,
)


_BUILD_DIR = Path(__file__).parent / "_wavesurfer_frontend"
_component = components.declare_component(
    "texture_map_region_wavesurfer",
    path=str(_BUILD_DIR),
)


def region_wavesurfer(
    audio_src: str,
    regions: list[dict[str, Any]],
    *,
    key: str,
    wave_options: WaveSurferOptions,
    region_colormap: str = "cool",
    show_controls: bool = False,
    plugins: list[str] | None = None,
) -> Any:
    """Render WaveSurfer and return region bounds after a drag or resize ends."""
    plugin_configurations = (
        DEFAULT_PLUGINS.to_dict()
        if plugins is None
        else WaveSurferPluginConfigurationList.from_name_list(plugins).to_dict()
    )

    return _component(
        audio_src=audio_to_base64(audio_src),
        regions=regions,
        key=key,
        default=None,
        wave_options=wave_options.to_dict(),
        region_colormap=region_colormap,
        controls=show_controls,
        plugin_configurations=plugin_configurations,
    )
