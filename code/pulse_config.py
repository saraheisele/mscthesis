"""Shared pulse-type configuration for the activity analysis pipeline.

Analysis part: configuration (used by preprocessing, plotting, and correlation scripts).
Dependencies: none.
"""

PULSE_TYPES = {
    "all": {
        "label": "all pulses",
        "suffix": "_all",
        "figures_subdir": "all_pulses",
        "hist_subdir": "all_pulses_hist",
        "array": None,
    },
    "double": {
        "label": "double pulses",
        "suffix": "_dp",
        "figures_subdir": "double_pulses",
        "hist_subdir": "double_pulses_hist",
        "array": "is_double_peak",
    },
    "wide": {
        "label": "wide pulses",
        "suffix": "_wide",
        "figures_subdir": "wide_pulses",
        "hist_subdir": "wide_pulses_hist",
        "array": "is_wide_pulse",
    },
}

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
