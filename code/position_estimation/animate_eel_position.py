"""Animate eel movement along the Berlin line logger for one wav chunk.

Analysis part: position visualization (Part 5c of Berlin activity analysis).
Dependencies: data_paths, position_utils, eelplotting.

Provide a synced eellogger wav file; loads matching predetected pulses from the
session h5, estimates head position per pulse, and renders an animation.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__)

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath
from scipy.io import wavfile

from data_paths import EEL_SVG, LAB_DATA_DIR, POSITION_FIGURES_DIR, PROJECT_ROOT
from eelplotting import get_eel_shape, plot_eel
from position_utils import (
    BRIGHT_POOL_CENTER,
    DARK_POOL_CENTER,
    DEFAULT_BRIGHT_DARK_BOUNDARY_M,
    LINE_LENGTH_M,
    POOL_RADIUS_M,
    default_electrode_positions_m,
    eel_body_endpoints,
    find_entry_recordings,
    fused_pool_outline_vertices,
    load_pulses_for_wav,
    movement_direction,
    smooth_positions,
    tank_outline_bounds_framed,
)
from presentation_style import (
    NON_PULSE_SHAPE_COLOR,
    apply_presentation_style,
    copy_thesis_asset,
    thesis_figure_path,
)

EEL_LINE_Y = 0.0
ELECTRODE_TICK_HALF_M = 0.08
ELECTRODE_LABEL_OFFSET_M = 0.14
RAW_WINDOW_SEC = 0.10
RAW_WINDOW_MS = RAW_WINDOW_SEC * 1000.0
FIG_WIDTH_IN = 14.0
POS_PANEL_WIDTH_RATIO = 2
RAW_PANEL_WIDTH_RATIO = 1
TITLE_PAD = 8
FIXED_BODY_LENGTH_M = 2.0
RAW_Y_MARGIN = 1.08
POOL_FRAME_PAD_M = 0.04
THESIS_ANIM_FRAME_COUNT = 24
THESIS_ANIM_POSTER_FRAME = 12
# Four frames from a short turnaround scene (1-based indices into exported frames).
THESIS_PRINT_KEYFRAME_INDICES = (1, 2, 3, 4)
# Approximate clip length burned into the exported thesis frames (ms).
# Updated when --make-print-figure rebuilds a short turnaround scene.
THESIS_ANIM_DURATION_MS = 20_000.0
LATEX_POSITION_FIGURES = PROJECT_ROOT / "docs" / "latex_thesis" / "figures" / "position_estimation"

# Public URL for the print-thesis QR code and Direct GIF link.
# Use the GitHub blob page (opens/plays in browser). raw.githubusercontent.com
# serves the same bytes as a download for large GIFs.
ANIMATION_SUPPLEMENT_URL = (
    "https://github.com/saraheisele/mscthesis/blob/master/"
    "docs/latex_thesis/figures/position_estimation/eel_position_animation.gif"
)
# Landing page with notes + MP4 link.
ANIMATION_SUPPLEMENT_LANDING = (
    "https://github.com/saraheisele/mscthesis/blob/master/"
    "docs/supplement/eel_position_animation.md"
)

# Thesis default: dark-area chunk used for eel_position_animation.{mp4,gif}.
DEFAULT_WAV_RELATIVE = Path(
    "recordings_2026-01-28_darkarea/eellogger02-20260128T163148.wav"
)
DEFAULT_WAV_PATH = LAB_DATA_DIR / DEFAULT_WAV_RELATIVE


def draw_electrode_ticks(ax, electrode_positions: np.ndarray):
    """Mark each electrode along the line with a tick and index label."""
    for index, pos in enumerate(electrode_positions):
        ax.plot(
            [pos, pos],
            [EEL_LINE_Y - ELECTRODE_TICK_HALF_M, EEL_LINE_Y + ELECTRODE_TICK_HALF_M],
            color="#333333",
            linewidth=1.2,
            zorder=4,
        )
        ax.text(
            pos,
            EEL_LINE_Y - ELECTRODE_LABEL_OFFSET_M,
            str(index),
            ha="center",
            va="top",
            fontsize=8,
            color="#333333",
            zorder=5,
        )


def draw_tank_background(ax, boundary_m: float = DEFAULT_BRIGHT_DARK_BOUNDARY_M):
    """Shade bright/dark tank regions and draw the fused pool outer outline only."""
    bright = Circle(
        BRIGHT_POOL_CENTER,
        POOL_RADIUS_M,
        fill=True,
        facecolor="#fff8dc",
        edgecolor="none",
        alpha=0.85,
        zorder=1,
    )
    dark = Circle(
        DARK_POOL_CENTER,
        POOL_RADIUS_M,
        fill=True,
        facecolor="#2f2f4f",
        edgecolor="none",
        alpha=0.35,
        zorder=1,
    )
    ax.add_patch(bright)
    ax.add_patch(dark)

    outline = fused_pool_outline_vertices(
        BRIGHT_POOL_CENTER, DARK_POOL_CENTER, POOL_RADIUS_M
    )
    outline_closed = np.vstack([outline, outline[:1]])
    ax.add_patch(
        PathPatch(
            MplPath(outline_closed),
            fill=False,
            edgecolor="black",
            linewidth=2,
            zorder=2,
        )
    )

    ax.plot(
        [0, LINE_LENGTH_M],
        [EEL_LINE_Y, EEL_LINE_Y],
        color="#444444",
        linewidth=1.5,
        linestyle="-",
        zorder=3,
    )
    ax.axvline(boundary_m, color="gray", linestyle="--", linewidth=1, zorder=3)
    electrode_positions = default_electrode_positions_m()
    draw_electrode_ticks(ax, electrode_positions)


def load_wav_segment(
    wav_path: Path,
    max_seconds: float | None = None,
    channel: int | None = None,
) -> tuple[np.ndarray, int]:
    """Load wav data; use ``channel`` (0-based) or first channel if omitted."""
    fs, data = wavfile.read(str(wav_path))
    if data.ndim > 1:
        ch = 0 if channel is None else int(np.clip(channel, 0, data.shape[1] - 1))
        data = data[:, ch]
    data = data.astype(np.float32)
    if data.size and np.max(np.abs(data)) > 0:
        data = data / np.max(np.abs(data))
    if max_seconds is not None:
        data = data[: int(max_seconds * fs)]
    return data, int(fs)


def draw_realistic_eel(
    ax,
    head_m: float,
    tail_m: float,
    direction: float,
    body_length_m: float,
    svg_path: Path = EEL_SVG,
):
    """Draw the SVG-based eel body along the electrode line. Returns created artists."""
    length_m = body_length_m
    x_center = np.linspace(-length_m, 0, 300)
    y_center = np.zeros_like(x_center)
    rotate = 180 if (not np.isnan(direction) and direction < 0) else 0

    eel_verts, _, _ = get_eel_shape(
        str(svg_path),
        x_center,
        y_center,
        length_m,
        rotate=rotate,
        headpos=(head_m, EEL_LINE_Y),
    )
    return plot_eel(ax, eel_verts[0], eel_verts[1], color=NON_PULSE_SHAPE_COLOR, alpha=0.95)


def build_animation(
    pulse_positions,
    meta,
    body_length_m: float = FIXED_BODY_LENGTH_M,
    smooth_window: int = 5,
    fps: int = 10,
    max_frames: int = 500,
    show_raw: bool = True,
    audio_time_offset_sec: float = 0.0,
):
    """Create a matplotlib FuncAnimation with optional raw-audio panel.

    ``audio_time_offset_sec`` is the absolute wav time corresponding to
    animation t=0 (needed when the scene was time-shifted).
    """
    body_length_m = FIXED_BODY_LENGTH_M
    if not pulse_positions:
        raise ValueError(
            f"No pulses found for wav window {meta['wav_start']} – {meta['wav_end']}"
        )

    wav_path = Path(meta["wav_path"])
    times = np.array([pulse.time_sec for pulse in pulse_positions])
    positions = np.array([pulse.head_m for pulse in pulse_positions])
    smoothed = smooth_positions(positions, window=smooth_window)
    directions = movement_direction(positions, window=smooth_window)
    head_channels = np.asarray(
        [pulse.head_channel for pulse in pulse_positions], dtype=int
    )

    if len(times) > max_frames:
        step = int(np.ceil(len(times) / max_frames))
        times = times[::step]
        positions = positions[::step]
        smoothed = smoothed[::step]
        directions = directions[::step]
        head_channels = head_channels[::step]

    duration = max(float(times[-1]), 0.1)
    interval_ms = int(1000 / fps)

    # Load the wav segment that matches the (possibly shifted) animation clock.
    fs_wav, wav_full = wavfile.read(str(wav_path))
    audio_fs = int(fs_wav)
    i0 = max(0, int(float(audio_time_offset_sec) * audio_fs))
    i1 = min(len(wav_full), i0 + int((duration + 1.0) * audio_fs))
    wav_data = wav_full[i0:i1]
    if wav_data.ndim == 1:
        wav_data = wav_data[:, np.newaxis]
    wav_data = wav_data.astype(np.float32)
    peak_all = float(np.max(np.abs(wav_data))) if wav_data.size else 1.0
    if peak_all > 0:
        wav_data = wav_data / peak_all
    audio_time_ms = np.arange(len(wav_data)) / audio_fs * 1000.0
    duration_ms = duration * 1000.0

    dominant_channel = (
        int(np.bincount(head_channels).argmax()) if head_channels.size else 0
    )
    global_raw_ymax = RAW_Y_MARGIN

    xmin, xmax, ymin, ymax = tank_outline_bounds_framed(POOL_FRAME_PAD_M)
    data_aspect = (xmax - xmin) / (ymax - ymin)
    pos_panel_width_in = FIG_WIDTH_IN * POS_PANEL_WIDTH_RATIO / (
        POS_PANEL_WIDTH_RATIO + RAW_PANEL_WIDTH_RATIO
    )
    fig_height_in = pos_panel_width_in / data_aspect + 0.8

    if show_raw:
        fig, (ax_pos, ax_raw) = plt.subplots(
            1,
            2,
            figsize=(FIG_WIDTH_IN, fig_height_in),
            gridspec_kw={
                "width_ratios": [POS_PANEL_WIDTH_RATIO, RAW_PANEL_WIDTH_RATIO],
                "wspace": 0.10,
            },
        )
    else:
        fig, ax_pos = plt.subplots(figsize=(pos_panel_width_in, fig_height_in))
        ax_raw = None

    fig.set_layout_engine(None)
    fig.suptitle(
        f"Eel movement — {wav_path.name}",
        y=0.97,
    )

    draw_tank_background(ax_pos)
    ax_pos.set_xlim(xmin, xmax)
    ax_pos.set_ylim(ymin, ymax)
    ax_pos.set_aspect("equal")
    ax_pos.margins(0)
    for spine in ax_pos.spines.values():
        spine.set_visible(False)
    ax_pos.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax_pos.set_xlabel("Position along electrode line", labelpad=2)
    ax_pos.set_title(
        f"{meta['wav_start'].strftime('%Y-%m-%d %H:%M:%S')} · "
        f"peak positive · {body_length_m:.1f} m eel",
        pad=TITLE_PAD,
    )

    raw_line = None
    if show_raw and ax_raw is not None:
        raw_line, = ax_raw.plot([], [], color="#555555", linewidth=1.8, alpha=0.95)
        ax_raw.set_ylabel("Normalized audio")
        ax_raw.set_xlabel("Time (ms)")
        ax_raw.set_title(
            f"Channel {dominant_channel} · "
            f"±{RAW_WINDOW_MS / 2:.0f} ms window",
            pad=TITLE_PAD,
        )
        ax_raw.grid(True, alpha=0.2)
        ax_raw.set_ylim(-global_raw_ymax, global_raw_ymax)
        ax_raw.spines["top"].set_visible(False)
        ax_raw.spines["right"].set_visible(False)

    if show_raw:
        fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.13, wspace=0.10)
    else:
        fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.13)

    trail_line, = ax_pos.plot([], [], color=NON_PULSE_SHAPE_COLOR, linewidth=2, alpha=0.6, zorder=4)
    eel_artists = []

    def init():
        trail_line.set_data([], [])
        if raw_line is not None:
            raw_line.set_data([], [])
        return [trail_line]

    def update_raw_window(t: float, channel: int):
        if raw_line is None:
            return
        half_ms = RAW_WINDOW_MS / 2.0
        t_ms = t * 1000.0
        t0 = max(0.0, t_ms - half_ms)
        t1 = min(duration_ms, t_ms + half_ms)
        if t1 - t0 < RAW_WINDOW_MS:
            if t0 == 0.0:
                t1 = min(duration_ms, RAW_WINDOW_MS)
            else:
                t0 = max(0.0, duration_ms - RAW_WINDOW_MS)
        mask = (audio_time_ms >= t0) & (audio_time_ms <= t1)
        ch = int(np.clip(channel, 0, wav_data.shape[1] - 1))
        y = wav_data[mask, ch]
        raw_line.set_data(audio_time_ms[mask], y)
        ax_raw.set_xlim(t0, t1)
        # Scale to this window so EODs are visible (global peak-norm hides them).
        local_peak = float(np.max(np.abs(y))) if y.size else 0.0
        ylim = max(local_peak * RAW_Y_MARGIN, 0.05)
        ax_raw.set_ylim(-ylim, ylim)
        ax_raw.set_title(
            f"Channel {ch} · ±{RAW_WINDOW_MS / 2:.0f} ms window",
            pad=TITLE_PAD,
        )

    def update(frame_idx):
        nonlocal eel_artists
        for artist in eel_artists:
            artist.remove()
        eel_artists = []

        t = times[frame_idx]
        head_m = smoothed[frame_idx]
        direction = directions[frame_idx]
        head_m, tail_m = eel_body_endpoints(head_m, direction, body_length_m=body_length_m)

        trail_line.set_data(smoothed[: frame_idx + 1], np.full(frame_idx + 1, EEL_LINE_Y))

        eel_artists.extend(
            draw_realistic_eel(ax_pos, head_m, tail_m, direction, body_length_m)
        )

        ax_pos.set_xlabel(
            f"Position along electrode line · "
            f"t = {t * 1000:.0f} ms / {duration_ms:.0f} ms"
        )
        update_raw_window(t, int(head_channels[frame_idx]))
        return [trail_line, *eel_artists]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(times),
        init_func=init,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    meta["n_anim_frames"] = len(times)
    meta["duration_sec"] = duration
    meta["audio_fs"] = audio_fs
    return fig, anim


def save_animation(
    fig,
    anim,
    output_path: Path,
    fps: int = 10,
    wav_path: Path | None = None,
    duration_sec: float | None = None,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.set_layout_engine("none")
    video_path = output_path
    if wav_path is not None and output_path.suffix.lower() == ".mp4":
        video_path = output_path.with_suffix(".video-only.mp4")

    if output_path.suffix.lower() == ".gif":
        anim.save(output_path, writer="pillow", fps=fps, dpi=150)
    else:
        anim.save(video_path, writer="ffmpeg", fps=fps, dpi=150)

    if wav_path is not None and output_path.suffix.lower() == ".mp4":
        import subprocess
        import tempfile

        audio_input = str(wav_path)
        temp_audio = None
        if duration_sec is not None:
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_audio.close()
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(wav_path),
                    "-t", str(duration_sec),
                    "-c:a", "pcm_s16le",
                    temp_audio.name,
                ],
                check=False,
                capture_output=True,
            )
            audio_input = temp_audio.name

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", audio_input,
                "-c:v", "copy", "-c:a", "aac",
                "-shortest",
                str(output_path),
            ],
            check=False,
            capture_output=True,
        )
        if temp_audio is not None:
            Path(temp_audio.name).unlink(missing_ok=True)
        if video_path != output_path and video_path.exists():
            video_path.unlink(missing_ok=True)


def _copy_into_latex_position_figures(source: Path, filename: str) -> Path:
    LATEX_POSITION_FIGURES.mkdir(parents=True, exist_ok=True)
    dest = LATEX_POSITION_FIGURES / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return dest


def save_thesis_animation_frames(
    fig,
    anim,
    *,
    n_total: int,
    n_frames: int = THESIS_ANIM_FRAME_COUNT,
    poster_frame: int = THESIS_ANIM_POSTER_FRAME,
) -> list[Path]:
    """Write evenly spaced PNG frames used by the thesis ``animategraphics`` figure."""
    n_total = int(n_total or getattr(anim, "save_count", 0) or 0)
    if n_total < 1:
        raise ValueError("Animation has no frames to export.")
    n_frames = min(int(n_frames), n_total)
    indices = np.unique(np.linspace(0, n_total - 1, n_frames).astype(int))
    while len(indices) < n_frames:
        extra = [i for i in range(n_total) if i not in set(indices)]
        if not extra:
            break
        indices = np.append(indices, extra[0])
    indices = np.sort(indices)[:n_frames]

    frame_dir = thesis_figure_path("position_estimation/anim_frames/frame_001.png").parent
    latex_frame_dir = LATEX_POSITION_FIGURES / "anim_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    latex_frame_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    fig.set_layout_engine("none")
    for out_idx, src_idx in enumerate(indices, start=1):
        anim._func(int(src_idx))
        name = f"frame_{out_idx:03d}.png"
        thesis_path = frame_dir / name
        fig.savefig(thesis_path, dpi=150, bbox_inches=None)
        shutil.copy2(thesis_path, latex_frame_dir / name)
        saved.append(thesis_path)

    poster_idx = min(max(int(poster_frame), 1), len(saved))
    still = saved[poster_idx - 1]
    still_name = "eel_position_still.png"
    copy_thesis_asset(still, f"position_estimation/{still_name}")
    _copy_into_latex_position_figures(still, still_name)
    print(f"Saved {len(saved)} thesis animation frames to {frame_dir}")
    return saved


def _anim_frame_paths(frame_dir: Path | None = None) -> list[Path]:
    frame_dir = Path(frame_dir) if frame_dir is not None else (
        LATEX_POSITION_FIGURES / "anim_frames"
    )
    paths = sorted(frame_dir.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"No animation frames found in {frame_dir}")
    return paths


def _crop_position_panel(image, *, full_frame: bool = True):
    """Optionally crop a thesis anim frame to the left (tank) panel."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    if full_frame:
        # Keep the full animation frame (tank + raw panel) so nothing is clipped.
        return image
    width, height = image.size
    # Layout matches build_animation: ~2:1 width split with a white gutter near x≈0.61.
    left = int(0.02 * width)
    right = int(0.585 * width)
    top = int(0.14 * height)
    # Drop the burned-in x-axis time string; panel labels carry the times.
    bottom = int(0.88 * height)
    return image.crop((left, top, right, bottom))


def _frame_time_label(
    frame_number_1based: int,
    n_frames: int = THESIS_ANIM_FRAME_COUNT,
    duration_ms: float | None = None,
) -> str:
    duration_ms = THESIS_ANIM_DURATION_MS if duration_ms is None else float(duration_ms)
    if n_frames <= 1:
        t_ms = 0.0
    else:
        t_ms = (frame_number_1based - 1) / (n_frames - 1) * duration_ms
    if t_ms >= 60_000:
        return f"t = {t_ms / 60_000:.1f} min"
    return f"t = {t_ms / 1000:.1f} s"


def save_animation_qr_code(
    url: str = ANIMATION_SUPPLEMENT_URL,
    *,
    output_name: str = "eel_position_animation_qr.png",
) -> Path:
    """Write a QR PNG that points to the electronic animation supplement."""
    import qrcode

    qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=12, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

    thesis_path = thesis_figure_path(f"position_estimation/{output_name}")
    img.save(thesis_path)
    _copy_into_latex_position_figures(thesis_path, output_name)
    print(f"Saved animation QR code → {url}")
    return thesis_path


def save_print_keyframe_figure(
    *,
    frame_dir: Path | None = None,
    frame_numbers: tuple[int, ...] = THESIS_PRINT_KEYFRAME_INDICES,
    output_name: str = "eel_position_keyframes.png",
    duration_ms: float | None = None,
    full_frame: bool = True,
    time_labels_ms: list[float] | tuple[float, ...] | None = None,
) -> Path:
    """Compose a print-friendly keyframe strip from exported anim frames."""
    from PIL import Image

    apply_presentation_style()
    available = {int(p.stem.split("_")[1]): p for p in _anim_frame_paths(frame_dir)}
    missing = [n for n in frame_numbers if n not in available]
    if missing:
        raise FileNotFoundError(f"Missing anim frames: {missing}")

    crops = []
    labels = []
    n_label_frames = len(frame_numbers)
    for i, n in enumerate(frame_numbers, start=1):
        crops.append(_crop_position_panel(Image.open(available[n]), full_frame=full_frame))
        if time_labels_ms is not None and i - 1 < len(time_labels_ms):
            t_ms = float(time_labels_ms[i - 1])
            if t_ms >= 60_000:
                labels.append(f"t = {t_ms / 60_000:.1f} min")
            else:
                labels.append(f"t = {t_ms / 1000:.1f} s")
        else:
            labels.append(
                _frame_time_label(i, n_frames=n_label_frames, duration_ms=duration_ms)
            )

    n_cols = 2 if len(crops) <= 4 else 3
    n_rows = int(np.ceil(len(crops) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.5, 3.4 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, crop, label, panel in zip(axes, crops, labels, "abcdef"):
        ax.imshow(crop)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(f"({panel})  {label}", fontsize=12, fontweight="bold", labelpad=4)
    for ax in axes[len(crops) :]:
        ax.set_axis_off()

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.10, wspace=0.04, hspace=0.18)
    thesis_path = thesis_figure_path(f"position_estimation/{output_name}")
    fig.savefig(thesis_path, dpi=200, bbox_inches="tight")
    _copy_into_latex_position_figures(thesis_path, output_name)
    processed = POSITION_FIGURES_DIR / "animations" / output_name
    processed.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(processed, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved print keyframe figure to {thesis_path}")
    return thesis_path


def find_turnaround_scene(
    times: np.ndarray,
    positions: np.ndarray,
    *,
    window_s: float = 40.0,
    min_pulses: int = 25,
    min_window_s: float = 10.0,
    max_window_s: float = 60.0,
) -> tuple[float, float]:
    """Return (t0, t1) for a short window with a clear direction reversal.

    Window length is constrained to ``[min_window_s, max_window_s]``.
    """
    sm = smooth_positions(positions, window=5)
    best = None
    t_start = float(times[0])
    t_end = float(times[-1])
    windows = [
        w
        for w in (window_s, 30.0, 20.0, 50.0, 15.0, 60.0, 10.0)
        if min_window_s - 1e-6 <= w <= max_window_s + 1e-6
    ]
    for win in windows:
        for t0 in np.arange(t_start, t_end - win, 0.5):
            mask = (times >= t0) & (times < t0 + win)
            if int(mask.sum()) < min_pulses:
                continue
            pp = sm[mask]
            for kind, i_ex in (("min", int(np.argmin(pp))), ("max", int(np.argmax(pp)))):
                frac = i_ex / len(pp)
                if not (0.25 <= frac <= 0.75):
                    continue
                left, right = pp[: i_ex + 1], pp[i_ex:]
                if left.size < 8 or right.size < 8:
                    continue
                if kind == "min":
                    if not (left[0] - left[-1] > 0.8 and right[-1] - right[0] > 0.8):
                        continue
                    travel = (left[0] - left.min()) + (right[-1] - right.min())
                else:
                    if not (left[-1] - left[0] > 0.8 and right[0] - right[-1] > 0.8):
                        continue
                    travel = (left.max() - left[0]) + (right.max() - right[-1])
                span = float(pp.max() - pp.min())
                score = travel * span
                if best is None or score > best[0]:
                    best = (score, float(t0), float(t0 + win))
        if best is not None:
            break
    if best is None:
        mid = 0.5 * (t_start + t_end)
        win = min(max(window_s, min_window_s), max_window_s)
        return mid - 0.5 * win, mid + 0.5 * win
    return best[1], best[2]


def _raw_window_segment(
    wav_data: np.ndarray,
    fs: int,
    t_abs: float,
    channel: int,
    *,
    window_sec: float = RAW_WINDOW_SEC,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_ms relative to pulse, y) for a fixed electrode channel."""
    half = window_sec / 2.0
    i0 = max(0, int((t_abs - half) * fs))
    i1 = min(len(wav_data), int((t_abs + half) * fs))
    ch = int(np.clip(channel, 0, wav_data.shape[1] - 1))
    y = wav_data[i0:i1, ch].astype(np.float64)
    t_ms = (np.arange(i0, i1) / fs - t_abs) * 1000.0
    return t_ms, y


def _pick_polarity_contrast_pulses(pulse_positions, wav_data, fs: int, raw_channel: int = 0):
    """Pick (bright-end, dark-end) pulses for the print polarity contrast figure.

    Panel A: head at electrode 0, facing toward electrode 0 (dir < 0), strong
    positive peak on ``raw_channel``.
    Panel B: head at electrode 15, facing toward electrode 15 (dir > 0),
    dominant negative peak on the same ``raw_channel``.
    """
    times = np.asarray([p.time_sec for p in pulse_positions], dtype=float)
    channels = np.asarray([p.head_channel for p in pulse_positions], dtype=int)
    positions = np.asarray([p.head_m for p in pulse_positions], dtype=float)
    smoothed = smooth_positions(positions, window=5)
    directions = movement_direction(positions, window=5)

    best_a = None
    for i, p in enumerate(pulse_positions):
        if channels[i] != 0:
            continue
        if not np.isfinite(directions[i]) or directions[i] >= 0:
            continue
        _, y = _raw_window_segment(wav_data, fs, times[i], raw_channel)
        if y.size == 0:
            continue
        peak = float(y.max())
        trough = float(y.min())
        if peak < 800:
            continue
        score = peak - 0.2 * abs(trough)
        if best_a is None or score > best_a[0]:
            best_a = (score, i, peak, trough)

    best_b = None
    for i, p in enumerate(pulse_positions):
        if channels[i] != 15:
            continue
        if not np.isfinite(directions[i]) or directions[i] <= 0:
            continue
        if smoothed[i] < 3.2:
            continue
        _, y = _raw_window_segment(wav_data, fs, times[i], raw_channel)
        if y.size == 0:
            continue
        peak = float(y.max())
        trough = float(y.min())
        if trough > -200:
            continue
        score = abs(trough) - 0.2 * peak
        if best_b is None or score > best_b[0]:
            best_b = (score, i, peak, trough)

    if best_a is None or best_b is None:
        raise RuntimeError(
            "Could not find polarity-contrast pulses "
            f"(A={best_a is not None}, B={best_b is not None})."
        )
    return (
        pulse_positions[best_a[1]],
        pulse_positions[best_b[1]],
        directions[best_a[1]],
        directions[best_b[1]],
        smoothed[best_a[1]],
        smoothed[best_b[1]],
    )


def save_polarity_contrast_figure(
    *,
    wav_path: Path | None = None,
    raw_channel: int = 0,
    output_name: str = "eel_position_keyframes.png",
    body_length_m: float = FIXED_BODY_LENGTH_M,
) -> Path:
    """Two vertical panels: head@electrode 0 vs head@electrode 15, same raw channel.

    Channel selection note: elsewhere the animation raw panel follows each pulse's
    ``head_channel`` (electrode with the strongest *positive* peak). This print
    figure instead fixes ``raw_channel`` (default 0 / electrode 0) in both panels
    so amplitude and polarity can be compared directly.
    """
    apply_presentation_style()
    wav_path = Path(wav_path) if wav_path is not None else DEFAULT_WAV_PATH
    pulse_positions, meta = load_pulses_for_wav(wav_path)
    fs, wav_data = wavfile.read(str(wav_path))
    if wav_data.ndim == 1:
        wav_data = wav_data[:, np.newaxis]

    pulse_a, pulse_b, dir_a, dir_b, sm_a, sm_b = _pick_polarity_contrast_pulses(
        pulse_positions, wav_data, int(fs), raw_channel=raw_channel
    )

    panels = [
        {
            "pulse": pulse_a,
            "direction": float(dir_a),
            "head_m": float(sm_a),
            "label": "(a)  Head at electrode 0, facing electrode 0",
        },
        {
            "pulse": pulse_b,
            "direction": float(dir_b),
            "head_m": float(sm_b),
            "label": "(b)  Head at electrode 15, facing electrode 15",
        },
    ]

    # Shared raw y-scale (honest amplitude comparison on the same electrode).
    segs = []
    for panel in panels:
        t_ms, y = _raw_window_segment(
            wav_data, int(fs), float(panel["pulse"].time_sec), raw_channel
        )
        segs.append((t_ms, y))
    peak = max(float(np.max(np.abs(y))) for _, y in segs if y.size) or 1.0
    ylim = peak * RAW_Y_MARGIN

    xmin, xmax, ymin, ymax = tank_outline_bounds_framed(POOL_FRAME_PAD_M)
    fig = plt.figure(figsize=(12.5, 8.6))
    # Two rows × (position | raw)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[POS_PANEL_WIDTH_RATIO, RAW_PANEL_WIDTH_RATIO],
        height_ratios=[1, 1],
        wspace=0.12,
        hspace=0.28,
        left=0.05,
        right=0.98,
        top=0.92,
        bottom=0.08,
    )

    for row, (panel, (t_ms, y)) in enumerate(zip(panels, segs)):
        ax_pos = fig.add_subplot(gs[row, 0])
        ax_raw = fig.add_subplot(gs[row, 1])
        draw_tank_background(ax_pos)
        ax_pos.set_xlim(xmin, xmax)
        ax_pos.set_ylim(ymin, ymax)
        ax_pos.set_aspect("equal")
        ax_pos.margins(0)
        for spine in ax_pos.spines.values():
            spine.set_visible(False)
        ax_pos.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        head_m = panel["head_m"]
        direction = panel["direction"]
        head_m, tail_m = eel_body_endpoints(head_m, direction, body_length_m=body_length_m)
        draw_realistic_eel(ax_pos, head_m, tail_m, direction, body_length_m)
        # Red marker at electrode 0 (shared raw channel; both panels).
        electrode_positions = default_electrode_positions_m()
        ax_pos.plot(
            electrode_positions[0],
            EEL_LINE_Y,
            marker="o",
            markersize=9,
            color="red",
            markeredgecolor="black",
            markeredgewidth=0.6,
            zorder=8,
            clip_on=False,
        )
        # Panel caption as xlabel; titles use presentation rcParams.
        ax_pos.set_xlabel(panel["label"])
        ax_pos.set_title(
            f"t = {panel['pulse'].time_sec:.1f} s",
            pad=TITLE_PAD,
        )

        ax_raw.plot(t_ms, y, color="#555555", linewidth=1.6, alpha=0.95)
        ax_raw.set_ylim(-ylim, ylim)
        ax_raw.set_xlim(-RAW_WINDOW_MS / 2.0, RAW_WINDOW_MS / 2.0)
        ax_raw.grid(True, alpha=0.25)
        ax_raw.spines["top"].set_visible(False)
        ax_raw.spines["right"].set_visible(False)
        ax_raw.set_title(
            f"Electrode {raw_channel} (shared) · ±{RAW_WINDOW_MS / 2:.0f} ms",
            pad=TITLE_PAD,
        )
        if row == 1:
            ax_raw.set_xlabel("Time relative to pulse (ms)")
        ax_raw.set_ylabel("Raw amplitude")

    # No figure-level title (avoids overlap with panel titles).
    thesis_path = thesis_figure_path(f"position_estimation/{output_name}")
    fig.savefig(thesis_path, dpi=200, bbox_inches="tight")
    _copy_into_latex_position_figures(thesis_path, output_name)
    processed = POSITION_FIGURES_DIR / "animations" / output_name
    processed.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(processed, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Saved polarity-contrast keyframes to {thesis_path} "
        f"(A t={pulse_a.time_sec:.2f}s ch={pulse_a.head_channel}; "
        f"B t={pulse_b.time_sec:.2f}s ch={pulse_b.head_channel}; "
        f"raw electrode {raw_channel})"
    )
    return thesis_path


def make_print_animation_assets(
    *,
    wav_path: Path | None = None,
    qr_url: str | None = None,
    n_print_frames: int = 4,
) -> tuple[Path, Path]:
    """Rebuild print keyframes (polarity contrast) and leave QR unchanged.

    GIF/QR from prior runs are kept as-is unless missing; this path only
    refreshes the two-panel print figure.
    """
    _ = (n_print_frames, qr_url)  # retained for call-site compatibility
    wav_path = Path(wav_path) if wav_path is not None else DEFAULT_WAV_PATH
    keyframes = save_polarity_contrast_figure(wav_path=wav_path, raw_channel=0)
    qr_path = thesis_figure_path("position_estimation/eel_position_animation_qr.png")
    if not qr_path.exists():
        qr_path = save_animation_qr_code(url=ANIMATION_SUPPLEMENT_URL)
    else:
        print(f"Left existing QR untouched: {qr_path}")
    return keyframes, qr_path


def _print_entry_candidates(limit: int = 20) -> list[dict]:
    """List entry-recording candidates and return the full candidate list."""
    candidates = find_entry_recordings()
    if not candidates:
        print("No entry recordings found.")
        return []
    print(f"Found {len(candidates)} candidate chunks:\n")
    for item in candidates[:limit]:
        print(
            f"  {item['session']} / {item['wav_name']} — "
            f"entry from {item['entry_from']}, "
            f"{item['early_mean_m']:.2f} m → {item['late_mean_m']:.2f} m "
            f"({item['n_pulses']} pulses)"
        )
        print(f"    {item['wav_path']}")
    return candidates


def prompt_for_wav_path(default: Path = DEFAULT_WAV_PATH) -> Path:
    """Ask whether to keep the default chunk or choose another wav."""
    print("Default animation chunk:")
    print(f"  {DEFAULT_WAV_RELATIVE}")
    print(f"  {default}")
    if not default.exists():
        print("  (warning: default wav file not found on disk)")

    answer = input("Continue with this default chunk? [Y/n]: ").strip().lower()
    if answer in ("", "y", "yes"):
        return default

    while True:
        choice = input(
            "Enter a wav path, or 'list' to show entry candidates: "
        ).strip()
        if not choice:
            print("No path entered; using default chunk.")
            return default
        if choice.lower() in ("list", "l"):
            candidates = _print_entry_candidates()
            if not candidates:
                continue
            print(
                "\nPaste one of the wav paths above, or enter another path."
            )
            continue
        path = Path(choice).expanduser()
        if path.exists():
            return path
        print(f"File not found: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Animate eel position along the Berlin line logger for one wav chunk."
    )
    parser.add_argument(
        "wav_path",
        nargs="?",
        help=(
            "Path to an eellogger wav file (5-min chunk). "
            "If omitted, prompts to keep the default chunk or choose another."
        ),
    )
    parser.add_argument(
        "--h5",
        type=Path,
        default=None,
        help="Optional explicit path to the session pulses h5 file.",
    )
    parser.add_argument(
        "--body-length",
        type=float,
        default=FIXED_BODY_LENGTH_M,
        help="Assumed eel body length in metres for drawing (fixed at 2.0 m).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Moving-average window for direction estimation (default: 5 pulses).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Animation frames per second (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (.gif or .mp4). Defaults to data/processed/position_analysis/animations/.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the animation interactively instead of only saving.",
    )
    parser.add_argument(
        "--list-entry-recordings",
        action="store_true",
        help="List h5 chunks where the eel appears to enter from an edge.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Save .mp4 without muxing the original wav audio track.",
    )
    parser.add_argument(
        "--no-raw-panel",
        action="store_true",
        help="Hide the raw audio / pulse marker panel.",
    )
    parser.add_argument(
        "--make-print-figure",
        action="store_true",
        help=(
            "Only compose the print keyframe strip + QR code from existing "
            "anim_frames (no wav / animation rebuild)."
        ),
    )
    return parser.parse_args()


def main():
    apply_presentation_style()
    args = parse_args()

    if args.make_print_figure:
        make_print_animation_assets()
        return

    if args.list_entry_recordings:
        _print_entry_candidates()
        return

    wav_path = Path(args.wav_path) if args.wav_path else prompt_for_wav_path()
    pulse_positions, meta = load_pulses_for_wav(wav_path, h5_path=args.h5)
    print(
        f"Loaded {len(pulse_positions)} pulses from {meta['h5_path']} "
        f"for wav window {meta['wav_start']} – {meta['wav_end']}"
    )

    fig, anim = build_animation(
        pulse_positions,
        meta,
        body_length_m=args.body_length,
        smooth_window=args.smooth_window,
        fps=args.fps,
        show_raw=not args.no_raw_panel,
    )

    use_audio = not args.no_audio
    if args.output is None:
        out_dir = POSITION_FIGURES_DIR / "animations"
        out_dir.mkdir(parents=True, exist_ok=True)
        primary_ext = ".mp4" if use_audio else ".gif"
        output_path = out_dir / f"{wav_path.stem}_peak_positive{primary_ext}"
        gif_path = out_dir / f"{wav_path.stem}_peak_positive.gif"
    else:
        output_path = Path(args.output)
        gif_path = (
            output_path.with_suffix(".gif")
            if output_path.suffix.lower() != ".gif"
            else None
        )

    save_thesis_animation_frames(fig, anim, n_total=int(meta["n_anim_frames"]))
    save_animation(
        fig,
        anim,
        output_path,
        fps=args.fps,
        wav_path=wav_path if use_audio and output_path.suffix.lower() == ".mp4" else None,
        duration_sec=meta.get("duration_sec"),
    )
    plt.close(fig)
    thesis_name = f"eel_position_animation{output_path.suffix}"
    copy_thesis_asset(output_path, f"position_estimation/{thesis_name}")
    _copy_into_latex_position_figures(output_path, thesis_name)
    print(f"Saved animation to {output_path}")

    # Also produce a silent GIF thesis asset when the primary output is MP4.
    if gif_path is not None and output_path.suffix.lower() == ".mp4" and output_path.exists():
        import subprocess

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(output_path),
                "-vf",
                f"fps={args.fps}",
                str(gif_path),
            ],
            check=False,
            capture_output=True,
        )
        if gif_path.exists():
            copy_thesis_asset(gif_path, "position_estimation/eel_position_animation.gif")
            _copy_into_latex_position_figures(
                gif_path, "eel_position_animation.gif"
            )
            print(f"Saved GIF animation to {gif_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
