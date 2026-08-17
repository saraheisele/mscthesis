"""Shared pulse-type configuration for the activity analysis pipeline.

Analysis part: configuration (used by preprocessing, plotting, and correlation scripts).
Dependencies: none.
"""

# Predetected ``raw_pulses`` snippets are stored on a common sample grid: 24 kHz
# recordings were interpolated up to the 48 kHz window length. Use this rate for
# every waveform sample↔time conversion (half-width, peak separation, plot axes,
# ms windows on snippets). Keep the file metadata ``samplerate`` only for
# recording-timeline quantities (``centers`` → wall-clock time, ISI between
# centers, recording duration).
WAVEFORM_FS = 48_000.0

PULSE_TYPES = {
    "all": {
        "label": "All pulses",
        "suffix": "_all",
        "figures_subdir": "all_pulses",
        "hist_subdir": "all_pulses_hist",
        "array": None,
    },
    "double": {
        "label": "Double pulses",
        "suffix": "_dp",
        "figures_subdir": "double_pulses",
        "hist_subdir": "double_pulses_hist",
        "array": "is_double_peak",
    },
    "wide": {
        "label": "Wide pulses",
        "suffix": "_wide",
        "figures_subdir": "wide_pulses",
        "hist_subdir": "wide_pulses_hist",
        "array": "is_wide_pulse",
    },
}

# Legend / panel order when several pulse types are drawn together.
PULSE_TYPE_DISPLAY_ORDER = ("all", "wide", "double")

SPECIAL_PULSE_TYPES = {
    key: value for key, value in PULSE_TYPES.items() if key != "all"
}

TIMESCALES = ["minute", "hour", "day", "month", "month_since_start", "year"]


def select_pulse_type(default="all", prompt="Pulse analysis type"):
    """Prompt user to select a pulse type key from PULSE_TYPES."""
    choices = ", ".join(PULSE_TYPES)
    selected = input(f"{prompt} ({choices}) [{default}]: ").strip().lower()
    if not selected:
        return default
    if selected not in PULSE_TYPES:
        raise ValueError(
            f"Unknown pulse analysis type '{selected}'. Choose one of: {choices}."
        )
    return selected
