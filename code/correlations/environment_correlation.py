"""Correlate eel pulse rate with water temperature and conductivity.

Analysis part: environmental correlation (Part 3 of Berlin activity analysis).
Dependencies: data_paths, pulse_config; requires eel_data_preprocessing.py output.

Loads Excel sensor logs, aligns daily/monthly pulse-rate histograms with
environmental averages, computes Pearson/Spearman correlations, and saves plots.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_setup import setup_script_paths

setup_script_paths(__file__)

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy import stats

from data_paths import ENVIRONMENT_CORRELATION_DIR, EXCEL_DIR, activity_hist_dir
from presentation_style import LEGEND_LOC, apply_presentation_style, pulse_shape_color, save_thesis_figure
from pulse_config import PULSE_TYPES

OUTPUT_DIR = ENVIRONMENT_CORRELATION_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_environmental_data():
    """
    Load and consolidate temperature and conductivity data from all Excel files.

    Returns:
        pd.DataFrame: Consolidated data with columns [date, temperature, conductivity]
                     Dates are normalized to datetime objects.
    """
    excel_files = sorted(EXCEL_DIR.glob("*Verlauf-relLeitwert*.xlsx"))
    print(f"Found {len(excel_files)} environmental data files")

    dfs = []
    for excel_file in excel_files:
        print(f"Loading {excel_file.name}...")
        df = pd.read_excel(excel_file, sheet_name=0)

        # Standardize column names
        date_col = "date" if "date" in df.columns else "Datum"
        temp_col = "Temperatur [°C]"
        cond_col = "Leitwert [µSiemens/cm]"

        # Select relevant columns
        df = df[[date_col, temp_col, cond_col]].copy()
        df.columns = ["date", "temperature", "conductivity"]

        # Parse dates - handle mixed formats
        df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")

        # Remove rows with NaN dates or missing values
        df = df.dropna(subset=["date", "temperature", "conductivity"])

        dfs.append(df)

    # Consolidate all files
    env_data = pd.concat(dfs, ignore_index=True)
    env_data = env_data.sort_values("date").reset_index(drop=True)

    # Remove duplicates (keep first occurrence)
    env_data = env_data.drop_duplicates(subset=["date"], keep="first")

    print(f"Consolidated environmental data: {len(env_data)} records")
    print(f"Date range: {env_data['date'].min()} to {env_data['date'].max()}")
    print(
        f"Temperature range: {env_data['temperature'].min():.1f}°C to {env_data['temperature'].max():.1f}°C"
    )
    print(
        f"Conductivity range: {env_data['conductivity'].min():.0f} to {env_data['conductivity'].max():.0f} µS/cm"
    )

    return env_data


def aggregate_daily_environmental(env_data):
    """
    Create daily averages of environmental parameters.

    Args:
        env_data (pd.DataFrame): Consolidated environmental data

    Returns:
        pd.DataFrame: Daily averages with date, temperature_daily, conductivity_daily
    """
    env_data["date_only"] = env_data["date"].dt.date
    daily = (
        env_data.groupby("date_only")
        .agg({"temperature": "mean", "conductivity": "mean"})
        .reset_index()
    )
    daily.columns = ["date", "temperature_daily", "conductivity_daily"]
    daily["date"] = pd.to_datetime(daily["date"])

    print(f"Created {len(daily)} daily environmental records")
    return daily


def aggregate_monthly_environmental(env_data):
    """
    Create monthly averages of environmental parameters.

    Args:
        env_data (pd.DataFrame): Consolidated environmental data

    Returns:
        pd.DataFrame: Monthly averages with year_month, temperature_monthly, conductivity_monthly
    """
    env_data["year_month"] = env_data["date"].dt.to_period("M")
    monthly = (
        env_data.groupby("year_month")
        .agg({"temperature": "mean", "conductivity": "mean"})
        .reset_index()
    )
    monthly.columns = ["year_month", "temperature_monthly", "conductivity_monthly"]
    # Convert period to first day of month for easier comparison
    monthly["date"] = monthly["year_month"].dt.start_time

    print(f"Created {len(monthly)} monthly environmental records")
    return monthly


#################################
############# LOAD PULSE DATA #############
#################################


def load_pulse_data(pulse_type="all"):
    """
    Load pulse rate histograms and metadata.

    Args:
        pulse_type (str): Type of pulses to load (all, double, wide, fat)

    Returns:
        tuple: (pulse_rate_dict, metadata, pulse_config)
    """
    hist_subdir = PULSE_TYPES[pulse_type]["hist_subdir"]
    data_path = activity_hist_dir(hist_subdir)

    metadata = np.load(data_path / "berlin_dummypulses_hist_metadata.npz")
    pulse_rate_dict = np.load(
        data_path / "berlin_dummypulses_pulse_rate_hz_hist_dict.npz"
    )

    print(f"Loaded {pulse_type} pulse data: {len(pulse_rate_dict.files)} timescales")

    return pulse_rate_dict, metadata, PULSE_TYPES[pulse_type]


#################################
############# ALIGNMENT #############
#################################


def align_daily_data(pulse_rate_dict, metadata, daily_env):
    """
    Align daily pulse rate with daily environmental data.

    Args:
        pulse_rate_dict: Loaded pulse rate histogram dict
        metadata: Metadata with first_year
        daily_env: Daily environmental data

    Returns:
        pd.DataFrame: Aligned daily data with pulse_rate_hz, temperature, conductivity
    """
    pulse_daily = pulse_rate_dict["day"]
    first_year = int(metadata["first_year"])

    # Create date array for daily pulse data
    # Pulse data: day of year (0-365)
    pulse_dates = []
    for year_offset in range(4):  # 4 years of data
        year = first_year + year_offset
        for day_idx in range(len(pulse_daily)):
            if not np.isnan(pulse_daily[day_idx]):
                date = datetime(year, 1, 1) + timedelta(days=day_idx)
                pulse_dates.append(
                    {"date": date, "pulse_rate_hz": pulse_daily[day_idx]}
                )

    pulse_df = pd.DataFrame(pulse_dates)
    pulse_df["date"] = pd.to_datetime(pulse_df["date"])

    # Merge with environmental data
    merged = pd.merge(pulse_df, daily_env, on="date", how="inner")

    print(f"Daily alignment: {len(merged)} matched records")

    return merged


def align_monthly_data(pulse_rate_dict, metadata, monthly_env):
    """
    Align monthly pulse rate with monthly environmental data.

    Args:
        pulse_rate_dict: Loaded pulse rate histogram dict
        metadata: Metadata with first_month_year, first_month_month
        monthly_env: Monthly environmental data

    Returns:
        pd.DataFrame: Aligned monthly data with pulse_rate_hz, temperature, conductivity
    """
    pulse_monthly = pulse_rate_dict["month_since_start"]
    first_month_year = int(metadata["first_month_year"])
    first_month_month = int(metadata["first_month_month"])

    # Create date array for monthly pulse data
    pulse_dates = []
    first_month = datetime(first_month_year, first_month_month, 1)
    for month_idx in range(len(pulse_monthly)):
        if not np.isnan(pulse_monthly[month_idx]):
            date = first_month + relativedelta(months=month_idx)
            pulse_dates.append(
                {"date": date, "pulse_rate_hz": pulse_monthly[month_idx]}
            )

    pulse_df = pd.DataFrame(pulse_dates)
    pulse_df["date"] = pd.to_datetime(pulse_df["date"])

    # Merge with environmental data
    merged = pd.merge(pulse_df, monthly_env, on="date", how="inner")

    print(f"Monthly alignment: {len(merged)} matched records")

    return merged


#################################
############# CORRELATIONS #############
#################################


def calculate_correlations(data_daily, data_monthly):
    """
    Calculate correlations between pulse rate and environmental parameters.

    Args:
        data_daily (pd.DataFrame): Daily aligned data
        data_monthly (pd.DataFrame): Monthly aligned data

    Returns:
        dict: Correlation results
    """
    results = {}

    # Daily correlations
    if len(data_daily) > 3:
        results["daily"] = {
            "n_samples": len(data_daily),
            "temperature": {
                "pearson": stats.pearsonr(
                    data_daily["temperature_daily"], data_daily["pulse_rate_hz"]
                ),
                "spearman": stats.spearmanr(
                    data_daily["temperature_daily"], data_daily["pulse_rate_hz"]
                ),
            },
            "conductivity": {
                "pearson": stats.pearsonr(
                    data_daily["conductivity_daily"], data_daily["pulse_rate_hz"]
                ),
                "spearman": stats.spearmanr(
                    data_daily["conductivity_daily"], data_daily["pulse_rate_hz"]
                ),
            },
        }

    # Monthly correlations
    if len(data_monthly) > 3:
        results["monthly"] = {
            "n_samples": len(data_monthly),
            "temperature": {
                "pearson": stats.pearsonr(
                    data_monthly["temperature_monthly"], data_monthly["pulse_rate_hz"]
                ),
                "spearman": stats.spearmanr(
                    data_monthly["temperature_monthly"], data_monthly["pulse_rate_hz"]
                ),
            },
            "conductivity": {
                "pearson": stats.pearsonr(
                    data_monthly["conductivity_monthly"], data_monthly["pulse_rate_hz"]
                ),
                "spearman": stats.spearmanr(
                    data_monthly["conductivity_monthly"], data_monthly["pulse_rate_hz"]
                ),
            },
        }

    return results


def calculate_lagged_shape_correlations(
    daily_env: pd.DataFrame,
    pulse_fractions_daily: pd.DataFrame,
    lags_days: tuple[int, ...] = (0, 1, 3, 7, 14, 30),
) -> pd.DataFrame:
    """Cross-correlate environmental changes with pulse-shape fractions at positive lags."""
    rows = []
    merged_base = pd.merge(
        pulse_fractions_daily,
        daily_env.rename(columns={"temperature_daily": "temperature", "conductivity_daily": "conductivity"}),
        on="date",
        how="inner",
    )
    for lag in lags_days:
        env_shifted = daily_env.copy()
        env_shifted["date"] = env_shifted["date"] + timedelta(days=lag)
        merged = pd.merge(
            pulse_fractions_daily,
            env_shifted.rename(columns={"temperature_daily": "temperature", "conductivity_daily": "conductivity"}),
            on="date",
            how="inner",
        )
        for shape_col in [c for c in merged.columns if c.endswith("_fraction") or c == "all_pulse_rate_hz"]:
            shape_name = (
                "all"
                if shape_col == "all_pulse_rate_hz"
                else shape_col.replace("_fraction", "")
            )
            for env_col in ("temperature", "conductivity"):
                if env_col not in merged.columns:
                    continue
                mask = merged[shape_col].notna() & merged[env_col].notna()
                x = merged.loc[mask, env_col].values
                y = merged.loc[mask, shape_col].values
                if len(x) < 5:
                    continue
                pr, pp = stats.pearsonr(x, y)
                sr, sp = stats.spearmanr(x, y)
                rows.append(
                    {
                        "lag_days": lag,
                        "pulse_shape": shape_name,
                        "env_param": env_col,
                        "n": len(x),
                        "pearson_r": pr,
                        "pearson_p": pp,
                        "spearman_r": sr,
                        "spearman_p": sp,
                    }
                )
    return pd.DataFrame(rows)


def load_daily_shape_fractions() -> pd.DataFrame:
    """Daily fraction of each special pulse shape from preprocessed histograms."""
    from pulse_config import PULSE_TYPES

    all_path = activity_hist_dir(PULSE_TYPES["all"]["hist_subdir"])
    all_counts = np.load(all_path / "berlin_dummypulses_count_hist_dict.npz")["day"].astype(float)
    all_rates = np.load(all_path / "berlin_dummypulses_pulse_rate_hz_hist_dict.npz")["day"]
    rows = []
    first_year = int(np.load(all_path / "berlin_dummypulses_hist_metadata.npz")["first_year"])

    for year_offset in range(4):
        year = first_year + year_offset
        for day_idx in range(len(all_counts)):
            total = all_counts[day_idx]
            if total <= 0 or np.isnan(total):
                continue
            date = datetime(year, 1, 1) + timedelta(days=day_idx)
            row = {
                "date": pd.Timestamp(date),
                "all_count": total,
                "all_pulse_rate_hz": all_rates[day_idx],
            }
            for ptype in ("double", "wide", "fat"):
                sub = np.load(
                    activity_hist_dir(PULSE_TYPES[ptype]["hist_subdir"])
                    / "berlin_dummypulses_count_hist_dict.npz"
                )["day"].astype(float)
                row[f"{ptype}_fraction"] = sub[day_idx] / total if total > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _plot_lagged_correlation_on_axis(ax, sub: pd.DataFrame, shape_order: list[str], env_param: str):
    for shape in shape_order:
        s = sub[sub["pulse_shape"] == shape].sort_values("lag_days")
        if s.empty:
            continue
        label = "all pulses" if shape == "all" else shape
        ax.plot(
            s["lag_days"],
            s["spearman_r"],
            marker="o",
            linewidth=2.5,
            label=label,
            color=pulse_shape_color(shape),
        )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Lag (days): env change → pulse response")
    ax.set_ylabel("Spearman ρ")
    ax.set_title(f"Delayed correlation: {env_param}")
    ax.legend(loc=LEGEND_LOC)
    ax.grid(True, alpha=0.3)


def plot_lagged_correlations(lag_df: pd.DataFrame, output_dir: Path):
    if lag_df.empty:
        return
    shape_order = ["all", "double", "wide", "fat"]
    env_params = sorted(lag_df["env_param"].unique())
    for env_param in env_params:
        sub = lag_df[lag_df["env_param"] == env_param]
        fig, ax = plt.subplots(figsize=(10, 5))
        _plot_lagged_correlation_on_axis(ax, sub, shape_order, env_param)
        plt.tight_layout()
        fig.savefig(output_dir / f"lagged_correlation_{env_param}.png", dpi=300)
        plt.close(fig)

    if len(env_params) >= 2:
        fig, axes = plt.subplots(len(env_params), 1, figsize=(10, 4.5 * len(env_params)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, env_param in zip(axes, env_params):
            sub = lag_df[lag_df["env_param"] == env_param]
            _plot_lagged_correlation_on_axis(ax, sub, shape_order, env_param)
        plt.tight_layout()
        save_thesis_figure("correlations/lag_corr_all_pulse_shapes.png", fig)
        plt.close(fig)


def run_lagged_environment_analysis(daily_env: pd.DataFrame):
    """Analyse delayed effects of temperature/conductivity on pulse shapes."""
    print("\n" + "=" * 70)
    print("LAGGED ENVIRONMENT → PULSE SHAPE CORRELATIONS")
    print("=" * 70)
    fractions = load_daily_shape_fractions()
    if fractions.empty:
        print("No daily shape fractions available.")
        return
    lag_df = calculate_lagged_shape_correlations(daily_env, fractions)
    lag_df.to_csv(OUTPUT_DIR / "lagged_env_pulse_shape_correlations.csv", index=False)
    plot_lagged_correlations(lag_df, OUTPUT_DIR)
    if not lag_df.empty:
        sig = lag_df[lag_df["spearman_p"] < 0.05].sort_values("spearman_r", key=abs, ascending=False)
        print(sig.head(10).to_string(index=False))


def print_correlation_summary(correlations, pulse_label):
    """Print formatted correlation results."""
    print(f"\n{'=' * 70}")
    print(f"CORRELATION ANALYSIS: {pulse_label}")
    print(f"{'=' * 70}")

    for timescale, corr_data in correlations.items():
        print(f"\n{timescale.upper()} DATA (n={corr_data['n_samples']}):")
        print("-" * 70)

        for param in ["temperature", "conductivity"]:
            pears_r, pears_p = corr_data[param]["pearson"]
            spear_rho, spear_p = corr_data[param]["spearman"]
            print(f"\n  {param.upper()}:")
            print(
                f"    Pearson:  r = {pears_r:7.3f}, p = {pears_p:.4f} {'***' if pears_p < 0.001 else '**' if pears_p < 0.01 else '*' if pears_p < 0.05 else ''}"
            )
            print(
                f"    Spearman: ρ = {spear_rho:7.3f}, p = {spear_p:.4f} {'***' if spear_p < 0.001 else '**' if spear_p < 0.01 else '*' if spear_p < 0.05 else ''}"
            )


def calculate_effect_size(r):
    """
    Interpret correlation effect size using Cohen's guidelines.

    Args:
        r (float): Correlation coefficient

    Returns:
        str: Effect size interpretation
    """
    abs_r = abs(r)
    if abs_r < 0.1:
        return "negligible"
    elif abs_r < 0.3:
        return "small"
    elif abs_r < 0.5:
        return "medium"
    else:
        return "large"


def print_all_pulses_detailed_summary(
    corr_all, data_daily_all, data_monthly_all, all_correlations
):
    """
    Print detailed summary specifically for overall activity (all pulses).

    Args:
        corr_all (dict): Correlations for all pulses
        data_daily_all (pd.DataFrame): Daily aligned data for all pulses
        data_monthly_all (pd.DataFrame): Monthly aligned data for all pulses
        all_correlations (dict): All correlations keyed by pulse type
    """
    print(f"\n{'#' * 70}")
    print("# DETAILED ANALYSIS: OVERALL ACTIVITY (ALL PULSES)")
    print(f"{'#' * 70}")

    # Summary statistics for all pulses
    print("\nSUMMARY STATISTICS FOR ALL PULSES:")
    print("-" * 70)

    if len(data_monthly_all) > 0:
        print("\nMonthly Activity Profile:")
        print(f"  Mean pulse rate: {data_monthly_all['pulse_rate_hz'].mean():.3f} Hz")
        print(f"  Std dev: {data_monthly_all['pulse_rate_hz'].std():.3f} Hz")
        print(
            f"  Range: {data_monthly_all['pulse_rate_hz'].min():.3f} - {data_monthly_all['pulse_rate_hz'].max():.3f} Hz"
        )

        print("\nWater Conditions Profile:")
        print(
            f"  Temperature: {data_monthly_all['temperature_monthly'].mean():.2f}°C (±{data_monthly_all['temperature_monthly'].std():.2f}°C)"
        )
        print(
            f"  Conductivity: {data_monthly_all['conductivity_monthly'].mean():.0f} µS/cm (±{data_monthly_all['conductivity_monthly'].std():.0f} µS/cm)"
        )

    # Detailed correlation results
    print(f"\n{'=' * 70}")
    print("CORRELATION WITH ENVIRONMENTAL PARAMETERS:")
    print(f"{'=' * 70}")

    for timescale in ["daily", "monthly"]:
        if timescale not in corr_all:
            continue

        data_key = (
            "temperature_daily" if timescale == "daily" else "temperature_monthly"
        )
        data = data_daily_all if timescale == "daily" else data_monthly_all

        print(f"\n{timescale.upper()} LEVEL (n={corr_all[timescale]['n_samples']}):")
        print("-" * 70)

        # Temperature
        pears_r, pears_p = corr_all[timescale]["temperature"]["pearson"]
        spear_rho, spear_p = corr_all[timescale]["temperature"]["spearman"]
        print("\nTEMPERATURE:")
        print(
            f"  Pearson:  r = {pears_r:7.3f}, p = {pears_p:.4f}, effect size: {calculate_effect_size(pears_r)}"
        )
        if pears_p < 0.05:
            print("    ✓ SIGNIFICANT (p < 0.05)")
        print(
            f"  Spearman: ρ = {spear_rho:7.3f}, p = {spear_p:.4f}, effect size: {calculate_effect_size(spear_rho)}"
        )
        if spear_p < 0.05:
            print("    ✓ SIGNIFICANT (p < 0.05)")

        # Conductivity
        pears_r, pears_p = corr_all[timescale]["conductivity"]["pearson"]
        spear_rho, spear_p = corr_all[timescale]["conductivity"]["spearman"]
        print("\nCONDUCTIVITY:")
        print(
            f"  Pearson:  r = {pears_r:7.3f}, p = {pears_p:.4f}, effect size: {calculate_effect_size(pears_r)}"
        )
        if pears_p < 0.05:
            print("    ✓ SIGNIFICANT (p < 0.05)")
        print(
            f"  Spearman: ρ = {spear_rho:7.3f}, p = {spear_p:.4f}, effect size: {calculate_effect_size(spear_rho)}"
        )
        if spear_p < 0.05:
            print("    ✓ SIGNIFICANT (p < 0.05)")

    # Comparison with pulse subtypes
    print(f"\n{'=' * 70}")
    print("COMPARISON WITH PULSE SUBTYPES:")
    print(f"{'=' * 70}")

    print("\nMonthly Conductivity Correlations (all pulse types):")
    print("-" * 70)

    for pulse_type in ["all", "double", "wide", "fat"]:
        if pulse_type in all_correlations and "monthly" in all_correlations[pulse_type]:
            corr_data = all_correlations[pulse_type]["monthly"]["conductivity"]
            spear_rho, spear_p = corr_data["spearman"]
            pears_r, pears_p = corr_data["pearson"]
            sig_flag = "✓" if spear_p < 0.05 else " "
            print(
                f"  {pulse_type.upper():8} | Spearman ρ = {spear_rho:7.3f} (p={spear_p:.4f}) {sig_flag}"
            )


def compare_pulse_types_contribution(all_correlations):
    """
    Show which pulse subtypes contribute most to overall activity patterns.

    Args:
        all_correlations (dict): Correlations for all pulse types
    """
    print(f"\n{'=' * 70}")
    print("PULSE TYPE CONTRIBUTION TO OVERALL ACTIVITY:")
    print(f"{'=' * 70}")

    print("\nMonthly Conductivity Response (correlation strength with activity):")
    print("-" * 70)

    pulse_order = []
    for pulse_type in ["all", "double", "wide", "fat"]:
        if pulse_type in all_correlations and "monthly" in all_correlations[pulse_type]:
            _, p = all_correlations[pulse_type]["monthly"]["conductivity"]["spearman"]
            r, _ = all_correlations[pulse_type]["monthly"]["conductivity"]["spearman"]
            pulse_order.append((pulse_type, r, p))

    # Sort by p-value and then by effect size
    pulse_order.sort(key=lambda x: (x[2], abs(x[1])))

    for pulse_type, rho, p in pulse_order:
        label = pulse_type.upper() if pulse_type != "all" else "ALL PULSES"
        sig = "✓ SIGNIFICANT" if p < 0.05 else "Not significant"
        print(f"  {label:12} | ρ = {rho:7.3f}, p = {p:.4f} | {sig}")


#################################
############# VISUALIZATION #############
#################################


def plot_timeseries(data_monthly, pulse_label):
    """
    Create a time series plot showing pulse rate, temperature, and conductivity together.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Pulse rate
    axes[0].plot(
        data_monthly["date"],
        data_monthly["pulse_rate_hz"],
        "o-",
        color="blue",
        linewidth=2,
        markersize=6,
        label="Pulse Rate",
    )
    axes[0].set_ylabel("Pulse Rate (Hz)", fontsize=11, fontweight="bold")
    axes[0].set_title(
        f"Eel Activity and Water Conditions - {pulse_label}",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    # Temperature
    axes[1].plot(
        data_monthly["date"],
        data_monthly["temperature_monthly"],
        "o-",
        color="red",
        linewidth=2,
        markersize=6,
        label="Temperature",
    )
    axes[1].set_ylabel("Temperature (°C)", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")

    # Conductivity
    axes[2].plot(
        data_monthly["date"],
        data_monthly["conductivity_monthly"],
        "o-",
        color="green",
        linewidth=2,
        markersize=6,
        label="Conductivity",
    )
    axes[2].set_ylabel("Conductivity (µS/cm)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Date", fontsize=11, fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"timeseries_{pulse_label.lower().replace(' ', '_')}.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved: timeseries_{pulse_label.lower().replace(' ', '_')}.png")
    plt.close()


def plot_correlations(data_daily, data_monthly, pulse_label):
    """
    Create scatter plots with regression lines showing correlations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color for daily/monthly points
    daily_color, monthly_color = "steelblue", "coral"

    # Daily Temperature
    if len(data_daily) > 1:
        z_temp_d = np.polyfit(
            data_daily["temperature_daily"], data_daily["pulse_rate_hz"], 1
        )
        p_temp_d = np.poly1d(z_temp_d)
        x_range = np.linspace(
            data_daily["temperature_daily"].min(),
            data_daily["temperature_daily"].max(),
            100,
        )
        axes[0].scatter(
            data_daily["temperature_daily"],
            data_daily["pulse_rate_hz"],
            alpha=0.6,
            s=50,
            color=daily_color,
            label="Daily",
        )
        axes[0].plot(x_range, p_temp_d(x_range), "--", color=daily_color, linewidth=2)

        r, p = stats.pearsonr(
            data_daily["temperature_daily"], data_daily["pulse_rate_hz"]
        )
        axes[0].text(
            0.05,
            0.95,
            f"r = {r:.3f}, p = {p:.4f}",
            transform=axes[0].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=10,
        )

    # Monthly Temperature
    if len(data_monthly) > 1:
        z_temp_m = np.polyfit(
            data_monthly["temperature_monthly"], data_monthly["pulse_rate_hz"], 1
        )
        p_temp_m = np.poly1d(z_temp_m)
        x_range = np.linspace(
            data_monthly["temperature_monthly"].min(),
            data_monthly["temperature_monthly"].max(),
            100,
        )
        axes[0].scatter(
            data_monthly["temperature_monthly"],
            data_monthly["pulse_rate_hz"],
            alpha=0.6,
            s=80,
            color=monthly_color,
            label="Monthly",
            marker="^",
        )
        axes[0].plot(x_range, p_temp_m(x_range), "--", color=monthly_color, linewidth=2)

        r, p = stats.pearsonr(
            data_monthly["temperature_monthly"], data_monthly["pulse_rate_hz"]
        )
        axes[0].text(
            0.05,
            0.88,
            f"r = {r:.3f}, p = {p:.4f}",
            transform=axes[0].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
            fontsize=10,
        )

    axes[0].set_xlabel("Temperature (°C)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Pulse Rate (Hz)", fontsize=11, fontweight="bold")
    axes[0].set_title("Pulse Rate vs Temperature", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Daily Conductivity
    if len(data_daily) > 1:
        z_cond_d = np.polyfit(
            data_daily["conductivity_daily"], data_daily["pulse_rate_hz"], 1
        )
        p_cond_d = np.poly1d(z_cond_d)
        x_range = np.linspace(
            data_daily["conductivity_daily"].min(),
            data_daily["conductivity_daily"].max(),
            100,
        )
        axes[1].scatter(
            data_daily["conductivity_daily"],
            data_daily["pulse_rate_hz"],
            alpha=0.6,
            s=50,
            color=daily_color,
            label="Daily",
        )
        axes[1].plot(x_range, p_cond_d(x_range), "--", color=daily_color, linewidth=2)

        r, p = stats.pearsonr(
            data_daily["conductivity_daily"], data_daily["pulse_rate_hz"]
        )
        axes[1].text(
            0.05,
            0.95,
            f"r = {r:.3f}, p = {p:.4f}",
            transform=axes[1].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=10,
        )

    # Monthly Conductivity
    if len(data_monthly) > 1:
        z_cond_m = np.polyfit(
            data_monthly["conductivity_monthly"], data_monthly["pulse_rate_hz"], 1
        )
        p_cond_m = np.poly1d(z_cond_m)
        x_range = np.linspace(
            data_monthly["conductivity_monthly"].min(),
            data_monthly["conductivity_monthly"].max(),
            100,
        )
        axes[1].scatter(
            data_monthly["conductivity_monthly"],
            data_monthly["pulse_rate_hz"],
            alpha=0.6,
            s=80,
            color=monthly_color,
            label="Monthly",
            marker="^",
        )
        axes[1].plot(x_range, p_cond_m(x_range), "--", color=monthly_color, linewidth=2)

        r, p = stats.pearsonr(
            data_monthly["conductivity_monthly"], data_monthly["pulse_rate_hz"]
        )
        axes[1].text(
            0.05,
            0.88,
            f"r = {r:.3f}, p = {p:.4f}",
            transform=axes[1].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
            fontsize=10,
        )

    axes[1].set_xlabel("Conductivity (µS/cm)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Pulse Rate (Hz)", fontsize=11, fontweight="bold")
    axes[1].set_title("Pulse Rate vs Conductivity", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle(
        f"Correlation Analysis - {pulse_label}", fontsize=14, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"correlations_{pulse_label.lower().replace(' ', '_')}.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved: correlations_{pulse_label.lower().replace(' ', '_')}.png")
    plt.close()


def plot_combined_correlations(aligned_by_pulse: dict):
    """Overlay all pulse-shape correlations on shared temperature/conductivity panels."""
    if "all" not in aligned_by_pulse:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    env_specs = [
        ("temperature_monthly", "Temperature (°C)"),
        ("conductivity_monthly", "Conductivity (µS/cm)"),
    ]

    for ax, (env_col, env_label) in zip(axes, env_specs):
        for pulse_type, payload in aligned_by_pulse.items():
            monthly = payload["monthly"]
            if len(monthly) < 2 or env_col not in monthly.columns:
                continue
            color = pulse_shape_color(pulse_type)
            label = PULSE_TYPES[pulse_type]["label"]
            ax.scatter(
                monthly[env_col],
                monthly["pulse_rate_hz"],
                alpha=0.65,
                s=70,
                color=color,
                label=label,
            )
            coeffs = np.polyfit(monthly[env_col], monthly["pulse_rate_hz"], 1)
            x_range = np.linspace(monthly[env_col].min(), monthly[env_col].max(), 100)
            ax.plot(x_range, np.poly1d(coeffs)(x_range), "--", color=color, linewidth=2.2)
        ax.set_xlabel(env_label)
        ax.set_ylabel("Pulse rate (Hz)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc=LEGEND_LOC)

    fig.suptitle("Environmental correlations — all pulse categories")
    plt.tight_layout()
    out = OUTPUT_DIR / "correlations_all_pulse_shapes.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    save_thesis_figure("correlations/correlations_all_pulse_shapes.png")
    print(f"Saved: {out.name}")
    plt.close()


def main():
    """Run temperature/conductivity correlation analysis for all pulse types."""
    apply_presentation_style()
    print("\n" + "=" * 70)
    print("CORRELATING EEL ACTIVITY WITH ENVIRONMENTAL PARAMETERS")
    print("=" * 70 + "\n")

    env_data = load_environmental_data()
    daily_env = aggregate_daily_environmental(env_data)
    monthly_env = aggregate_monthly_environmental(env_data)

    all_correlations = {}
    all_pulses_data = {}
    aligned_by_pulse = {}

    for pulse_type, pulse_config in PULSE_TYPES.items():
        print(f"\n{'─' * 70}")
        print(f"Loading {pulse_type} pulse data...")
        pulse_rate_dict, metadata, _ = load_pulse_data(pulse_type)

        data_daily = align_daily_data(pulse_rate_dict, metadata, daily_env)
        data_monthly = align_monthly_data(pulse_rate_dict, metadata, monthly_env)

        if len(data_monthly) > 3:
            correlations = calculate_correlations(data_daily, data_monthly)
            print_correlation_summary(correlations, pulse_config["label"])
            all_correlations[pulse_type] = correlations
            aligned_by_pulse[pulse_type] = {
                "daily": data_daily,
                "monthly": data_monthly,
            }

            if pulse_type == "all":
                all_pulses_data = {
                    "daily": data_daily,
                    "monthly": data_monthly,
                    "correlations": correlations,
                }

            plot_correlations(data_daily, data_monthly, pulse_config["label"])
        else:
            print(f"Skipping {pulse_type}: insufficient aligned data points")

    if all_pulses_data:
        print_all_pulses_detailed_summary(
            all_pulses_data["correlations"],
            all_pulses_data["daily"],
            all_pulses_data["monthly"],
            all_correlations,
        )
        compare_pulse_types_contribution(all_correlations)
        plot_combined_correlations(aligned_by_pulse)

    run_lagged_environment_analysis(daily_env)

    print(f"\nOutput saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
