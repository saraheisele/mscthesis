"""Compare special-pulse proportions in bright vs dark tank areas.

Analysis part: spatial pulse-shape comparison (Part 3c).
Dependencies: volley_pulse_analysis.load_raw_pulse_data, data_paths.

Uses filename tags (*_brightarea_* / *_darkarea_*) to group recordings and
tests whether double/wide pulse fractions differ between tank areas.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__)

import numpy as np
from scipy import stats

from data_paths import H5_ROOT
from volley_pulse_analysis import load_raw_pulse_data

def parse_tank_area(h5_file_path):
    """
    Infer tank area from recording filename.

    Berlin tank recordings use ``*_brightarea_*`` or ``*_darkarea_*`` suffixes.
    """
    name = h5_file_path.name.lower()
    if "brightarea" in name:
        return "bright"
    if "darkarea" in name:
        return "dark"
    return None


def accumulate_area_pulse_stats(h5_files):
    """
    Count pulse-type markers separately for bright- and dark-area recordings.

    Returns:
        dict: Area stats with pulse counts, totals, and number of files per area
    """
    area_stats = {
        "bright": {"double": 0, "wide": 0, "total": 0, "n_files": 0},
        "dark": {"double": 0, "wide": 0, "total": 0, "n_files": 0},
        "unlabeled": {"double": 0, "wide": 0, "total": 0, "n_files": 0},
    }

    for h5_file in h5_files:
        area = parse_tank_area(h5_file)
        area_key = area if area is not None else "unlabeled"

        pulse_centers, pulse_markers, _ = load_raw_pulse_data(h5_file)
        if pulse_centers is None or len(pulse_centers) == 0:
            continue

        area_stats[area_key]["n_files"] += 1
        area_stats[area_key]["total"] += len(pulse_centers)

        if pulse_markers:
            for ptype in ["double", "wide"]:
                if ptype in pulse_markers:
                    area_stats[area_key][ptype] += int(np.sum(pulse_markers[ptype]))

    return area_stats


def compare_area_pulse_type_proportions(area_stats, reference_area="bright"):
    """
    Compare pulse-type proportions between tank areas using two-proportion z-tests.

    Args:
        area_stats (dict): Output of accumulate_area_pulse_stats
        reference_area (str): Baseline area for comparison (default: bright)

    Returns:
        dict: Comparison statistics per pulse type and comparison area
    """
    comparison_areas = [
        area for area in ["bright", "dark"] if area != reference_area
    ]
    results = {}
    ref_total = area_stats[reference_area]["total"]
    ref_counts = area_stats[reference_area]

    for area in comparison_areas:
        area_total = area_stats[area]["total"]
        area_counts = area_stats[area]

        for ptype in ["double", "wide"]:
            ref_n = ref_counts[ptype]
            area_n = area_counts[ptype]

            if ref_total == 0 or area_total == 0:
                z_stat, p_value = np.nan, np.nan
            else:
                p_pool = (ref_n + area_n) / (ref_total + area_total)
                se = np.sqrt(
                    p_pool * (1 - p_pool) * (1 / ref_total + 1 / area_total)
                )
                if se == 0:
                    z_stat, p_value = np.nan, np.nan
                else:
                    p_ref = ref_n / ref_total
                    p_area = area_n / area_total
                    z_stat = (p_area - p_ref) / se
                    p_value = 2 * stats.norm.sf(abs(z_stat))

            results[(area, ptype)] = {
                "reference_pct": 100 * ref_n / ref_total if ref_total else np.nan,
                "area_pct": 100 * area_n / area_total if area_total else np.nan,
                "z_stat": z_stat,
                "p_value": p_value,
            }

    return results


def print_area_pulse_analysis(area_stats, reference_area="bright"):
    """Print formatted bright vs dark area pulse-type comparison."""
    print(f"\n{'=' * 70}")
    print("PULSE SHAPE vs TANK AREA (BRIGHT / DARK)")
    print(f"{'=' * 70}")

    print("\nLabeled recordings:")
    print("-" * 70)
    for area in ["bright", "dark"]:
        stats = area_stats[area]
        print(
            f"  {area.upper():5} area: {stats['n_files']:3d} files, "
            f"{stats['total']:6d} pulses"
        )

    unlabeled = area_stats["unlabeled"]
    if unlabeled["n_files"] > 0:
        print(
            f"  Unlabeled (no bright/dark in filename): {unlabeled['n_files']} files "
            f"({unlabeled['total']} pulses) — excluded from comparison"
        )

    ref_stats = area_stats[reference_area]
    if ref_stats["total"] == 0:
        print(f"\nNo {reference_area}-area pulses found. Skipping area comparison.")
        return

    print(f"\nPulse composition by area (n={ref_stats['total']} {reference_area} pulses):")
    print("-" * 70)
    for area in ["bright", "dark"]:
        total = area_stats[area]["total"]
        if total == 0:
            print(f"\n{area.upper()} AREA: no pulses")
            continue

        print(f"\n{area.upper()} AREA (n={total}):")
        for ptype in ["double", "wide"]:
            count = area_stats[area][ptype]
            pct = 100 * count / total
            print(f"  {ptype.upper():6}: {count:5d} pulses ({pct:5.1f}%)")

    comparisons = compare_area_pulse_type_proportions(
        area_stats, reference_area=reference_area
    )
    print(f"\nComparisons vs {reference_area.upper()} area:")
    print("-" * 70)

    for area in ["bright", "dark"]:
        if area == reference_area or area_stats[area]["total"] == 0:
            continue

        print(f"\n{area.upper()} vs {reference_area.upper()}:")
        for ptype in ["double", "wide"]:
            comp = comparisons[(area, ptype)]
            sig = ""
            if not np.isnan(comp["p_value"]) and comp["p_value"] < 0.05:
                direction = (
                    "higher"
                    if comp["area_pct"] > comp["reference_pct"]
                    else "lower"
                )
                sig = f" * significantly {direction} in {area} (p={comp['p_value']:.4f})"
            print(
                f"  {ptype.upper():6}: {comp['area_pct']:5.1f}% in {area}"
                f" vs {comp['reference_pct']:5.1f}% in {reference_area}{sig}"
            )



def run_area_analysis(h5_files=None):
    """Compare pulse-type proportions between bright and dark tank areas."""
    if h5_files is None:
        h5_files = sorted(H5_ROOT.glob("**/*.h5"))

    if not h5_files:
        print("No .h5 files found. Skipping area analysis.")
        return

    print(f"\n{'=' * 70}")
    print("PULSE SHAPE vs TANK AREA (BRIGHT / DARK)")
    print(f"{'=' * 70}")
    area_stats = accumulate_area_pulse_stats(h5_files)
    print_area_pulse_analysis(area_stats)


def main():
    run_area_analysis()


if __name__ == "__main__":
    main()
