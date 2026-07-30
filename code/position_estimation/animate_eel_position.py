"""Animate eel movement along the Berlin line logger for one wav chunk.

Analysis part: position visualization (Part 5c of Berlin activity analysis).
Dependencies: data_paths, position_utils, eelplotting.

Provide a synced eellogger wav file; loads matching predetected pulses from the
session h5, estimates head position per pulse, and renders an animation.
"""

from __future__ import annotations

import argparse
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

from data_paths import EEL_SVG, POSITION_FIGURES_DIR
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
from presentation_style import apply_presentation_style, copy_thesis_asset, pulse_shape_color

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


def load_wav_segment(wav_path: Path, max_seconds: float | None = None) -> tuple[np.ndarray, int]:
    """Load mono wav data (first channel if multichannel)."""
    fs, data = wavfile.read(str(wav_path))
    if data.ndim > 1:
        data = data[:, 0]
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
    return plot_eel(ax, eel_verts[0], eel_verts[1], color=pulse_shape_color("normal"), alpha=0.95)


def build_animation(
    pulse_positions,
    meta,
    body_length_m: float = FIXED_BODY_LENGTH_M,
    smooth_window: int = 5,
    fps: int = 10,
    max_frames: int = 500,
    show_raw: bool = True,
):
    """Create a matplotlib FuncAnimation with optional raw-audio panel."""
    body_length_m = FIXED_BODY_LENGTH_M
    if not pulse_positions:
        raise ValueError(
            f"No pulses found for wav window {meta['wav_start']} – {meta['wav_end']}"
        )

    times = np.array([pulse.time_sec for pulse in pulse_positions])
    positions = np.array([pulse.head_m for pulse in pulse_positions])
    smoothed = smooth_positions(positions, window=smooth_window)
    directions = movement_direction(positions, window=smooth_window)

    if len(times) > max_frames:
        step = int(np.ceil(len(times) / max_frames))
        times = times[::step]
        positions = positions[::step]
        smoothed = smoothed[::step]
        directions = directions[::step]

    duration = max(float(times[-1]), 0.1)
    interval_ms = int(1000 / fps)

    wav_path = Path(meta["wav_path"])
    raw_audio, audio_fs = load_wav_segment(wav_path, max_seconds=duration + 1)
    audio_time_ms = np.arange(len(raw_audio)) / audio_fs * 1000.0
    duration_ms = duration * 1000.0

    dominant_channel = (
        int(np.bincount([pulse.head_channel for pulse in pulse_positions]).argmax())
        if pulse_positions
        else 0
    )
    abs_audio = np.abs(raw_audio)
    peak_audio = float(np.max(abs_audio)) if abs_audio.size else 0.01
    global_raw_ymax = max(peak_audio * RAW_Y_MARGIN, 0.01)

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
        fontsize=13,
        fontweight="bold",
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
    ax_pos.set_xlabel("position along electrode line (m)", labelpad=2)
    ax_pos.set_title(
        f"{meta['wav_start'].strftime('%Y-%m-%d %H:%M:%S')} · "
        f"peak positive · {body_length_m:.1f} m eel",
        fontsize=11,
        pad=TITLE_PAD,
    )

    raw_line = None
    if show_raw and ax_raw is not None:
        raw_line, = ax_raw.plot([], [], color="#555555", linewidth=1.8, alpha=0.95)
        ax_raw.set_ylabel("normalized audio")
        ax_raw.set_xlabel("time (ms)")
        ax_raw.set_title(
            f"Channel {dominant_channel + 1} · "
            f"±{RAW_WINDOW_MS / 2:.0f} ms window",
            fontsize=11,
            pad=TITLE_PAD,
        )
        ax_raw.grid(True, alpha=0.2)
        ax_raw.set_ylim(-global_raw_ymax, global_raw_ymax)

    if show_raw:
        fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.13, wspace=0.10)
    else:
        fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.13)

    trail_line, = ax_pos.plot([], [], color=pulse_shape_color("all"), linewidth=2, alpha=0.6, zorder=4)
    eel_artists = []

    def init():
        trail_line.set_data([], [])
        if raw_line is not None:
            raw_line.set_data([], [])
        return [trail_line]

    def update_raw_window(t: float):
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
        raw_line.set_data(audio_time_ms[mask], raw_audio[mask])
        ax_raw.set_xlim(t0, t1)

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
            f"position along electrode line (m) · "
            f"t = {t * 1000:.0f} ms / {duration_ms:.0f} ms"
        )
        update_raw_window(t)
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

    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Animate eel position along the Berlin line logger for one wav chunk."
    )
    parser.add_argument(
        "wav_path",
        nargs="?",
        help="Path to an eellogger wav file (5-min chunk).",
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
    return parser.parse_args()


def main():
    apply_presentation_style()
    args = parse_args()

    if args.list_entry_recordings:
        candidates = find_entry_recordings()
        if not candidates:
            print("No entry recordings found.")
            return
        print(f"Found {len(candidates)} candidate chunks:\n")
        for item in candidates[:20]:
            print(
                f"  {item['session']} / {item['wav_name']} — "
                f"entry from {item['entry_from']}, "
                f"{item['early_mean_m']:.2f} m → {item['late_mean_m']:.2f} m "
                f"({item['n_pulses']} pulses)"
            )
            print(f"    {item['wav_path']}")
        return

    if not args.wav_path:
        raise SystemExit("Provide a wav_path or use --list-entry-recordings.")

    wav_path = Path(args.wav_path)
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

    save_animation(
        fig,
        anim,
        output_path,
        fps=args.fps,
        wav_path=wav_path if use_audio and output_path.suffix.lower() == ".mp4" else None,
        duration_sec=meta.get("duration_sec"),
    )
    thesis_name = f"eel_position_animation{output_path.suffix}"
    copy_thesis_asset(output_path, f"position_estimation/{thesis_name}")
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
            print(f"Saved GIF animation to {gif_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
