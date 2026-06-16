"""Animate eel movement along the Berlin line logger for one wav chunk.

Analysis part: position visualization (Part 5c of Berlin activity analysis).
Dependencies: data_paths, position_utils.

Provide a synced eellogger wav file; loads matching predetected pulses from the
session h5, estimates head position per pulse, and renders an animation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow

from data_paths import POSITION_FIGURES_DIR
from position_utils import (
    DEFAULT_BRIGHT_DARK_BOUNDARY_M,
    LINE_LENGTH_M,
    eel_body_endpoints,
    find_entry_recordings,
    load_pulses_for_wav,
    movement_direction,
    smooth_positions,
)


def draw_tank_background(ax, boundary_m: float = DEFAULT_BRIGHT_DARK_BOUNDARY_M):
    """Shade bright and dark tank regions along the line."""
    ax.axvspan(0, boundary_m, color="#fff8dc", alpha=0.8, label="bright area")
    ax.axvspan(boundary_m, LINE_LENGTH_M, color="#2f2f4f", alpha=0.35, label="dark area")
    ax.axvline(boundary_m, color="gray", linestyle="--", linewidth=1)


def draw_eel(ax, head_m: float, tail_m: float, direction: float):
    """Draw a simple eel body segment and head marker. Returns created artists."""
    body_y = 0.5
    artists = []
    body = ax.plot(
        [tail_m, head_m],
        [body_y, body_y],
        color="#1a535c",
        linewidth=10,
        solid_capstyle="round",
        zorder=3,
    )
    head = ax.scatter([head_m], [body_y], s=120, color="#ff6b6b", edgecolors="black", zorder=4)
    artists.extend(body)
    artists.append(head)

    if not np.isnan(direction) and direction != 0:
        arrow = FancyArrow(
            head_m,
            body_y + 0.15,
            0.25 * direction,
            0,
            width=0.04,
            length_includes_head=True,
            color="#ff6b6b",
            zorder=5,
        )
        ax.add_patch(arrow)
        artists.append(arrow)

    return artists


def build_animation(
    pulse_positions,
    meta,
    body_length_m: float = 2.0,
    smooth_window: int = 5,
    fps: int = 10,
    max_frames: int = 500,
):
    """Create a matplotlib FuncAnimation for one wav chunk."""
    if not pulse_positions:
        raise ValueError(
            f"No pulses found for wav window {meta['wav_start']} – {meta['wav_end']}"
        )

    times = np.array([pulse.time_sec for pulse in pulse_positions])
    positions = np.array([pulse.head_m for pulse in pulse_positions])
    smoothed = smooth_positions(positions, window=smooth_window)
    directions = movement_direction(positions, window=smooth_window)

    # Subsample if too many pulses for a manageable animation
    if len(times) > max_frames:
        step = int(np.ceil(len(times) / max_frames))
        times = times[::step]
        positions = positions[::step]
        smoothed = smoothed[::step]
        directions = directions[::step]

    duration = max(float(times[-1]), 0.1)
    interval_ms = int(1000 / fps)

    fig, ax = plt.subplots(figsize=(14, 4))
    draw_tank_background(ax)
    ax.set_xlim(-0.1, LINE_LENGTH_M + 0.1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("position along electrode line (m)")
    ax.set_yticks([])
    ax.set_title(
        f"Eel movement — {Path(meta['wav_path']).name}\n"
        f"{meta['wav_start'].strftime('%Y-%m-%d %H:%M:%S')} "
        f"({meta['method']}, body length {body_length_m:.1f} m)"
    )

    trail_line, = ax.plot([], [], color="#4ecdc4", linewidth=2, alpha=0.6, zorder=2)
    eel_artists = []

    def init():
        trail_line.set_data([], [])
        return [trail_line]

    def update(frame_idx):
        nonlocal eel_artists
        for artist in eel_artists:
            artist.remove()
        eel_artists = []

        t = times[frame_idx]
        head_m = smoothed[frame_idx]
        direction = directions[frame_idx]
        head_m, tail_m = eel_body_endpoints(head_m, direction, body_length_m=body_length_m)

        trail_line.set_data(smoothed[: frame_idx + 1], np.full(frame_idx + 1, 0.5))
        eel_artists.extend(draw_eel(ax, head_m, tail_m, direction))

        ax.set_xlabel(
            f"position along electrode line (m) — t = {t:.1f} s / {duration:.1f} s"
        )
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
    return fig, anim


def save_animation(fig, anim, output_path: Path, fps: int = 10):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".gif":
        anim.save(output_path, writer="pillow", fps=fps, dpi=150)
    else:
        anim.save(output_path, writer="ffmpeg", fps=fps, dpi=150)
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
        "--method",
        choices=("peak_positive", "weighted_mean"),
        default="peak_positive",
        help="Head position estimation method (default: peak_positive).",
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
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_entry_recordings:
        candidates = find_entry_recordings(method=args.method)
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
    pulse_positions, meta = load_pulses_for_wav(
        wav_path, h5_path=args.h5, method=args.method
    )
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
    )

    if args.output is None:
        out_dir = POSITION_FIGURES_DIR / "animations"
        out_name = f"{wav_path.stem}_{args.method}.gif"
        output_path = out_dir / out_name
    else:
        output_path = args.output

    save_animation(fig, anim, output_path, fps=args.fps)
    print(f"Saved animation to {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
