This directory contains a patched build derived from
[`streamlit-wavesurfer` 0.6.0](https://github.com/burstMembrane/streamlit_wavesurfer),
copyright 2025 Liam Power and distributed under the MIT License.

TextureMap's patch sends the current region bounds to Streamlit when a WaveSurfer
region drag or resize finishes. The upstream 0.6.0 build only sends the initial
region values.
