"""Shared helpers for parsing lab session Word notes and aligning recordings.

Analysis part: session-note utilities (shared by feeding and mating correlation).
Dependencies: data_paths; optional nixio / python-docx when reading .h5 / .docx.

Provides date/time parsing, docx text extraction, eellogger-on timestamps,
session↔docx lookup, and recording-start inference from h5 metadata + wav filenames.
Feeding- and mating-specific event extractors live in their own modules and call
into these helpers.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import nixio
from docx import Document

from data_paths import LAB_DATA_DIR

TIME_RE = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b")
DATE_INLINE_RE = re.compile(r"\b(20\d{6})\b")
SESSION_DATE_RE = re.compile(
    r"recordings_(\d{4}-\d{2}-\d{2})(?:_(\d{2})-(\d{2})-(\d{2}))?"
)
WAV_TIME_RE = re.compile(r"(\d{8})T(\d{6})")

# Minutes of leeway when scoring timed events against a candidate recording window
# (same radius used for feeding flags in correlate_activity_with_feeding).
FEEDING_FLAG_RADIUS_MIN = 5


def parse_session_date(session_name: str) -> datetime | None:
    match = SESSION_DATE_RE.match(session_name)
    if not match:
        return None
    date_str = match.group(1)
    if match.group(2):
        return datetime.strptime(
            f"{date_str} {match.group(2)}:{match.group(3)}:{match.group(4)}",
            "%Y-%m-%d %H:%M:%S",
        )
    return datetime.strptime(date_str, "%Y-%m-%d")


def parse_time_on_date(time_str: str, base_date: datetime) -> datetime | None:
    parts = [int(x) for x in time_str.split(":")]
    if len(parts) == 2:
        hour, minute = parts
        second = 0
    else:
        hour, minute, second = parts
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return None
    return base_date.replace(hour=hour, minute=minute, second=second, microsecond=0)


def docx_text(docx_path: Path) -> str:
    doc = Document(docx_path)
    chunks = [para.text for para in doc.paragraphs if para.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            chunks.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(chunks)


def extract_eellogger_on_times(session_name: str, docx_path: Path) -> list[datetime]:
    text = docx_text(docx_path)
    default_date = parse_session_date(session_name)
    if default_date is None:
        return []

    current_date = default_date.replace(hour=0, minute=0, second=0, microsecond=0)
    on_times = []

    for line in text.splitlines():
        inline_date = DATE_INLINE_RE.search(line.replace(" ", ""))
        if inline_date:
            try:
                current_date = datetime.strptime(inline_date.group(1), "%Y%m%d")
            except ValueError:
                pass

        if not re.search(r"e+e?llogger\s+on", line, re.IGNORECASE):
            continue
        for time_str in TIME_RE.findall(line):
            dt = parse_time_on_date(time_str, current_date)
            if dt is not None:
                on_times.append(dt)
    return sorted(set(on_times))


def find_docx_for_session(session_name: str) -> Path | None:
    images_dir = LAB_DATA_DIR / session_name / "images"
    if not images_dir.exists():
        return None
    docx_files = sorted(images_dir.glob("*.docx"))
    return docx_files[0] if docx_files else None


def session_name_from_h5(h5_path: Path) -> str:
    return h5_path.stem.replace("_pulses", "")


def get_wav_times(session_name: str) -> list[datetime]:
    session_dir = LAB_DATA_DIR / session_name
    if not session_dir.exists():
        base_date = parse_session_date(session_name)
        if base_date is None:
            return []
        session_dir = LAB_DATA_DIR / session_name.split("_")[0]
        if not session_dir.exists():
            parent = SESSION_DATE_RE.match(session_name)
            if parent:
                session_dir = LAB_DATA_DIR / f"recordings_{parent.group(1)}"
    if not session_dir.exists():
        return []

    wav_times = []
    for wav_path in session_dir.glob("eellogger*.wav"):
        match = WAV_TIME_RE.search(wav_path.name)
        if match:
            wav_times.append(
                datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
            )
    return sorted(wav_times)


def infer_recording_start(
    h5_path: Path,
    feeding_events: list[dict] | None = None,
    eellogger_on_times: list[datetime] | None = None,
    *,
    duration: float | None = None,
    h5_start: datetime | None = None,
) -> tuple[datetime, float, str]:
    session_name = session_name_from_h5(h5_path)
    filename_start = parse_session_date(session_name)

    if duration is None or h5_start is None:
        with nixio.File.open(str(h5_path)) as nix_file:
            meta = nix_file.sections["pulses_metadata"]["metadata"]
            duration = float(meta["duration"])
            h5_start = datetime.strptime(
                meta["metadata"]["INFO"]["DateTimeOriginal"], "%Y-%m-%dT%H:%M:%S"
            )

    if filename_start and SESSION_DATE_RE.match(session_name).group(2):
        return filename_start, duration, "h5_filename"

    wav_times = get_wav_times(session_name)
    feeding_events = feeding_events or []
    eellogger_on_times = eellogger_on_times or []

    def score_candidate(candidate: datetime) -> tuple:
        wav_diff = (
            min(abs((w - candidate).total_seconds()) for w in wav_times)
            if wav_times
            else 99999.0
        )
        rec_end = candidate + timedelta(seconds=duration)
        n_feeding = sum(
            1
            for event in feeding_events
            if candidate - timedelta(minutes=FEEDING_FLAG_RADIUS_MIN)
            <= event["event_time"]
            <= rec_end + timedelta(minutes=FEEDING_FLAG_RADIUS_MIN)
        )
        eel_diff = (
            min(abs((t - candidate).total_seconds()) for t in eellogger_on_times)
            if eellogger_on_times
            else 99999.0
        )
        return (wav_diff > 300, -n_feeding, wav_diff, eel_diff)

    candidates = [h5_start, h5_start - timedelta(hours=12), h5_start + timedelta(hours=12)]
    best = min(candidates, key=score_candidate)
    wav_diff = (
        min(abs((w - best).total_seconds()) for w in wav_times) if wav_times else float("inf")
    )
    if wav_diff <= 300:
        return best, duration, "wav_aligned"
    return h5_start, duration, "h5_metadata"
