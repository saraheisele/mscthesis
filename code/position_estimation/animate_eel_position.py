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
    eel_body_endpoints,
    find_entry_recordings,
    fused_pool_outline_vertices,
    load_pulses_for_wav,
    movement_direction,
    smooth_positions,
    tank_plot_limits,
)

EEL_LINE_Y = 0.0
RAW_WINDOW_SEC = 0.10
POS_PANEL_IN = 12.0
RAW_PANEL_WIDTH_RATIO = 1.6
FIG_HEIGHT_IN = POS_PANEL_IN
FIG_WIDTH_WITH_RAW_IN = POS_PANEL_IN * (1.0 + RAW_PANEL_WIDTH_RATIO)
FIG_WIDTH_POSITION_ONLY_IN = POS_PANEL_IN


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
        label="brightarea electrode line (3.75 m)",
    )
    ax.axvline(boundary_m, color="gray", linestyle="--", linewidth=1, zorder=3)
    ax.scatter(
        [0, LINE_LENGTH_M],
        [EEL_LINE_Y, EEL_LINE_Y],
        s=30,
        color="#333333",
        zorder=4,
        label="electrode 1 / 16",
    )


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
    length_m = max(abs(head_m - tail_m), body_length_m * 0.5, 0.4)
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
    return plot_eel(ax, eel_verts[0], eel_verts[1], color="#1a535c", alpha=0.95)


def build_animation(
    pulse_positions,
    meta,
    body_length_m: float = 2.0,
    smooth_window: int = 5,
    fps: int = 10,
    max_frames: int = 500,
    show_raw: bool = True,
):
    """Create a matplotlib FuncAnimation with optional raw-audio panel."""
    if not pulse_positions:
        raise ValueError(
            f"No pulses found for wav window {meta['wav_start']} – {meta['wav_end']}"
        )

    times = np.array([pulse.time_sec for pulse in pulse_positions])
    positions = np.array([pulse.head_m for pulse in pulse_positions])
    amplitudes = np.array([pulse.amplitude for pulse in pulse_positions])
    smoothed = smooth_positions(positions, window=smooth_window)
    directions = movement_direction(positions, window=smooth_window)
    amp_norm = amplitudes / (np.max(amplitudes) + 1e-12)

    if len(times) > max_frames:
        step = int(np.ceil(len(times) / max_frames))
        times = times[::step]
        positions = positions[::step]
        amplitudes = amplitudes[::step]
        amp_norm = amp_norm[::step]
        smoothed = smoothed[::step]
        directions = directions[::step]

    duration = max(float(times[-1]), 0.1)
    interval_ms = int(1000 / fps)

    wav_path = Path(meta["wav_path"])
    raw_audio, audio_fs = load_wav_segment(wav_path, max_seconds=duration + 1)
    audio_time = np.arange(len(raw_audio)) / audio_fs

    if show_raw:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(FIG_WIDTH_WITH_RAW_IN, FIG_HEIGHT_IN),
            gridspec_kw={"width_ratios": [1.0, RAW_PANEL_WIDTH_RATIO], "wspace": 0.06},
        )
        ax_pos, ax_raw = axes
        fig.subplots_adjust(left=0.04, right=0.99, top=0.90, bottom=0.12, wspace=0.06)
    else:
        fig, ax_pos = plt.subplots(figsize=(FIG_WIDTH_POSITION_ONLY_IN, FIG_HEIGHT_IN))
        ax_raw = None
        fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.12)

    xmin, xmax, ymin, ymax = tank_plot_limits()
    draw_tank_background(ax_pos)
    ax_pos.set_xlim(xmin, xmax)
    ax_pos.set_ylim(ymin, ymax)
    ax_pos.set_aspect("equal", adjustable="box")
    ax_pos.set_box_aspect(1)
    ax_pos.set_xlabel("position along electrode line (m)")
    ax_pos.set_yticks([])
    ax_pos.set_title(
        f"Eel movement — {wav_path.name}\n"
        f"{meta['wav_start'].strftime('%Y-%m-%d %H:%M:%S')} "
        f"(peak positive, body length {body_length_m:.1f} m)"
    )
    ax_pos.legend(loc="upper right", fontsize=8)

    raw_line = None
    if show_raw and ax_raw is not None:
        raw_line, = ax_raw.plot([], [], color="#555555", linewidth=1.0, alpha=0.95)
        ax_raw.set_ylabel("normalized audio")
        ax_raw.set_xlabel("time (s)")
        ax_raw.set_title(f"Raw recording (±{RAW_WINDOW_SEC / 2:.2f} s window, synced)")
        ax_raw.grid(True, alpha=0.2)

    trail_line, = ax_pos.plot([], [], color="#4ecdc4", linewidth=2, alpha=0.6, zorder=4)
    pulse_scatter = ax_pos.scatter(
        [],
        [],
        s=[],
        c=[],
        cmap="Reds",
        vmin=0,
        vmax=1,
        alpha=0.7,
        zorder=4,
        edgecolors="none",
    )
    eel_artists = []

    def init():
        trail_line.set_data([], [])
        pulse_scatter.set_offsets(np.empty((0, 2)))
        pulse_scatter.set_sizes([])
        pulse_scatter.set_array(np.array([]))
        if raw_line is not None:
            raw_line.set_data([], [])
        return [trail_line, pulse_scatter]

    def update_raw_window(t: float):
        if raw_line is None:
            return
        half = RAW_WINDOW_SEC / 2.0
        t0 = max(0.0, t - half)
        t1 = min(duration, t + half)
        if t1 - t0 < RAW_WINDOW_SEC:
            if t0 == 0.0:
                t1 = min(duration, RAW_WINDOW_SEC)
            else:
                t0 = max(0.0, duration - RAW_WINDOW_SEC)
        mask = (audio_time >= t0) & (audio_time <= t1)
        raw_line.set_data(audio_time[mask], raw_audio[mask])
        ax_raw.set_xlim(t0, t1)
        visible = raw_audio[mask]
        ymax = float(np.max(np.abs(visible)) * 1.25) if visible.size else 0.1
        ax_raw.set_ylim(-max(ymax, 0.05), max(ymax, 0.05))

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
        past = times <= t
        if past.any():
            past_amp = amp_norm[past]
            pulse_scatter.set_offsets(
                np.column_stack([positions[past], np.full(past.sum(), EEL_LINE_Y)])
            )
            pulse_scatter.set_sizes(20 + 200 * past_amp)
            pulse_scatter.set_array(past_amp)
        else:
            pulse_scatter.set_offsets(np.empty((0, 2)))
            pulse_scatter.set_sizes([])
            pulse_scatter.set_array(np.array([]))

        eel_artists.extend(
            draw_realistic_eel(ax_pos, head_m, tail_m, direction, body_length_m)
        )

        ax_pos.set_xlabel(
            f"position along electrode line (m) — t = {t:.1f} s / {duration:.1f} s"
        )
        update_raw_window(t)
        return [trail_line, pulse_scatter, *eel_artists]

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
        default=2.0,
        help="Assumed eel body length in metres for drawing (default: 2.0 = male).",
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
        ext = ".mp4" if use_audio else ".gif"
        out_name = f"{wav_path.stem}_peak_positive{ext}"
        output_path = out_dir / out_name
    else:
        output_path = args.output

    save_animation(
        fig,
        anim,
        output_path,
        fps=args.fps,
        wav_path=wav_path if use_audio and output_path.suffix.lower() == ".mp4" else None,
        duration_sec=meta.get("duration_sec"),
    )
    print(f"Saved animation to {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
