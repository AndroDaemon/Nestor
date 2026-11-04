"""Shared constants for the Nestor project."""

# Target sample rate for ASR (GigaAM expects 16 kHz)
SAMPLE_RATE = 16000

# Maximum chunk length for GigaAM (hard limit ~25 s, we stay below)
MAX_CHUNK_SECONDS = 25.0

# Default chunk length used in the pipeline
DEFAULT_CHUNK_SECONDS = 20.0