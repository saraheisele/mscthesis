"""
Correlate eel activity (pulse rate) with environmental parameters (temperature, conductivity).

This script:
1. Loads temperature and conductivity data from multiple Excel files
2. Consolidates and aligns environmental data with pulse rate histograms
3. Calculates correlations at daily and monthly timescales
4. Generates visualizations showing relationships between activity and environment

Data Sources:
    - Input: Excel files with environmental measurements (Verlauf-relLeitwert_*.xlsx)
    - Input: Pulse rate histograms from preprocessing (.npz files)
    - Output: Correlation matrices, statistical summaries, and visualization plots

Timescale Alignment:
    - Daily: Aligns daily pulse rates with daily environmental averages
    - Monthly: Aligns monthly pulse rates (month_since_start) with monthly environmental averages
"""

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy import stats
from scipy.signal import find_peaks
import warnings
import nixio

warnings.filterwarnings("ignore")

#################################
############# CONFIG #############
#################################

# Data paths
EXCEL_DIR = Path("/home/eisele/wrk/mscthesis/code")
PULSE_DATA_PATH = Path(
    "/home/eisele/wrk/mscthesis/data/intermediate/eels-mfn2021_dummy_activity_histograms/all_pulses_hist"
)
OUTPUT_DIR = Path("/home/eisele/wrk/mscthesis/data/processed/environment_correlation")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PULSE_TYPES = {
    "all": {"hist_subdir": "all_pulses_hist", "label": "All pulses"},
    "double": {"hist_subdir": "double_pulses_hist", "label": "Double pulses"},
    "wide": {"hist_subdir": "wide_pulses_hist", "label": "Wide pulses"},
    "fat": {"hist_subdir": "fat_pulses_hist", "label": "Fat pulses"},
}


#################################
############# LOAD ENV DATA #############
#################################


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
    data_path = Path(
        f"/home/eisele/wrk/mscthesis/data/intermediate/eels-mfn2021_dummy_activity_histograms/{hist_subdir}"
    )

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


#################################
############# VOLLEY ANALYSIS #############
#################################

VOLLEY_MIN_PULSES = 5
VOLLEY_MAX_ISI_MS = 2.0
VOLLEY_FALLBACK_MIN_PULSES = 3
VOLLEY_FALLBACK_MAX_ISI_MS = 5.0
VOLLEY_CLUSTER_PREFILTER_MS = 15.0
VOLLEY_SUBPEAK_MERGE_MS = 0.5
VOLLEY_CONTEXT_WINDOW_S = 5.0


def load_raw_pulse_data(h5_file_path):
    """
    Load pulse centers and markers from predetected .h5 file.

    Uses the same nixio layout as eel_data_preprocessing.py:
    block ``pulses`` for arrays, section ``pulses_metadata`` for samplerate.

    Args:
        h5_file_path (Path): Path to .h5 file

    Returns:
        tuple: (pulse_centers, pulse_markers_dict, metadata)
               - pulse_centers: Array of pulse center sample indices (predicted positive only)
               - pulse_markers_dict: Dict with marker arrays (double, wide, fat)
               - metadata: (sampling_rate, duration, start_time)
    """
    try:
        with nixio.File.open(str(h5_file_path)) as nix_file:
            if "pulses" not in [b.name for b in nix_file.blocks]:
                return None, None, None

            block = nix_file.blocks["pulses"]
            data_array_names = [da.name for da in block.data_arrays]
            if "centers" not in data_array_names:
                return None, None, None

            centers = block.data_arrays["centers"][:]
            if "predicted_labels" in data_array_names:
                positive_mask = block.data_arrays["predicted_labels"][:] == 1
            else:
                positive_mask = np.ones(len(centers), dtype=bool)

            pulse_centers = centers[positive_mask]
            pulse_markers = {}
            marker_map = {
                "is_double_peak": "double",
                "is_wide_pulse": "wide",
                "is_fat_pulse": "fat",
            }
            for array_name, marker_key in marker_map.items():
                if array_name in data_array_names:
                    pulse_markers[marker_key] = (
                        block.data_arrays[array_name][:][positive_mask].astype(bool)
                    )

            meta_section = nix_file.sections["pulses_metadata"]["metadata"]
            sampling_rate = float(meta_section["samplerate"])
            duration = float(meta_section["duration"])
            start_time_str = meta_section["metadata"]["INFO"]["DateTimeOriginal"]

            return (
                pulse_centers,
                pulse_markers,
                (sampling_rate, duration, start_time_str),
            )
    except Exception as e:
        print(f"Error loading {h5_file_path}: {e}")
        return None, None, None


def _get_representative_waveform(pulse_waveform):
    """Strongest channel waveform (same idea as double_peaks_detection.py)."""
    channel_strength = np.max(np.abs(pulse_waveform), axis=0)
    best_channel = int(np.argmax(channel_strength))
    return pulse_waveform[:, best_channel]


def _merge_event_times(event_times, sampling_rate, tolerance_ms=VOLLEY_SUBPEAK_MERGE_MS):
    if len(event_times) == 0:
        return event_times
    event_times = np.sort(event_times)
    tolerance_samples = int(sampling_rate * tolerance_ms / 1000)
    merged = [int(event_times[0])]
    for sample in event_times[1:]:
        if int(sample) - merged[-1] > tolerance_samples:
            merged.append(int(sample))
    return np.array(merged, dtype=np.int64)


def _find_short_isi_clusters(pulse_centers, sampling_rate, max_isi_ms=VOLLEY_CLUSTER_PREFILTER_MS):
    """Groups of consecutive detected pulses with inter-center ISI <= max_isi_ms."""
    if len(pulse_centers) < 2:
        return []

    pulse_centers = np.sort(pulse_centers)
    position_indices = np.arange(len(pulse_centers))
    isis = np.diff(pulse_centers) / sampling_rate * 1000
    clusters = []
    current = [0]

    for i, isi in enumerate(isis):
        if isi <= max_isi_ms:
            current.append(i + 1)
        else:
            if len(current) >= 2:
                clusters.append(position_indices[current])
            current = [i + 1]

    if len(current) >= 2:
        clusters.append(position_indices[current])

    return clusters


def _cluster_waveform_event_times(
    raw_pulses, all_centers, cluster_positions, positive_indices, sampling_rate
):
    """
    Build a max-amplitude envelope across overlapping snippets in a short-ISI cluster,
    then detect sub-peaks on that envelope.
    """
    half_width = len(raw_pulses[int(positive_indices[0])][:]) // 2
    min_distance = max(1, int(sampling_rate * 0.0005))
    envelope = {}

    for position in cluster_positions:
        pulse_idx = int(positive_indices[position])
        try:
            signal = np.abs(_get_representative_waveform(raw_pulses[pulse_idx][:]))
        except KeyError:
            continue
        base_sample = int(all_centers[pulse_idx]) - half_width
        for offset, amplitude in enumerate(signal):
            sample = base_sample + offset
            envelope[sample] = max(envelope.get(sample, 0.0), amplitude)

    if not envelope:
        return np.array([], dtype=np.int64)

    samples = np.array(sorted(envelope.keys()))
    amplitudes = np.array([envelope[sample] for sample in samples])
    peak_max = amplitudes.max()
    peak_indices, _ = find_peaks(
        amplitudes, height=peak_max * 0.25, distance=min_distance
    )
    return samples[peak_indices]


def extract_waveform_event_times(h5_file_path):
    """
    Derive sub-pulse event times from raw waveform snippets in short-ISI clusters.

    For consecutive detected pulses closer than VOLLEY_CLUSTER_PREFILTER_MS, overlapping
    snippets are merged into one envelope and sub-peaks are detected on that envelope.
    """
    try:
        with nixio.File.open(str(h5_file_path)) as nix_file:
            block = nix_file.blocks["pulses"]
            data_array_names = [da.name for da in block.data_arrays]
            if "raw_pulses" not in data_array_names or "centers" not in data_array_names:
                return None, None

            raw_pulses = block.data_arrays["raw_pulses"]
            all_centers = block.data_arrays["centers"][:]
            if "predicted_labels" in data_array_names:
                positive_indices = np.where(block.data_arrays["predicted_labels"][:] == 1)[0]
            else:
                positive_indices = np.arange(len(all_centers))

            if len(positive_indices) == 0:
                return np.array([], dtype=np.int64), None

            sampling_rate = float(
                nix_file.sections["pulses_metadata"]["metadata"]["samplerate"]
            )
            positive_centers = all_centers[positive_indices]
            clusters = _find_short_isi_clusters(positive_centers, sampling_rate)
            event_times = []

            for cluster in clusters:
                event_times.extend(
                    _cluster_waveform_event_times(
                        raw_pulses,
                        all_centers,
                        cluster,
                        positive_indices,
                        sampling_rate,
                    )
                )

            return _merge_event_times(
                np.array(event_times, dtype=np.int64), sampling_rate
            ), sampling_rate
    except Exception as e:
        print(f"Error extracting waveform events from {h5_file_path}: {e}")
        return None, None


def detect_volleys(
    event_times,
    sampling_rate,
    min_pulses=VOLLEY_MIN_PULSES,
    max_isi_ms=VOLLEY_MAX_ISI_MS,
):
    """
    Detect volleys from a sorted timeline of event sample indices.

    A volley is a run of at least ``min_pulses`` consecutive events where every
    inter-event interval is <= ``max_isi_ms``.
    """
    if event_times is None or len(event_times) < min_pulses:
        return []

    event_times = np.sort(event_times)
    isis = np.diff(event_times) / sampling_rate * 1000

    volleys = []
    current_volley = [0]

    for i, isi in enumerate(isis):
        if isi <= max_isi_ms:
            current_volley.append(i + 1)
        else:
            if len(current_volley) >= min_pulses:
                event_indices = np.array(current_volley)
                volleys.append(
                    {
                        "event_indices": event_indices,
                        "start_sample": int(event_times[event_indices[0]]),
                        "end_sample": int(event_times[event_indices[-1]]),
                        "n_events": len(event_indices),
                        "start_time_s": event_times[event_indices[0]] / sampling_rate,
                        "end_time_s": event_times[event_indices[-1]] / sampling_rate,
                    }
                )
            current_volley = [i + 1]

    if len(current_volley) >= min_pulses:
        event_indices = np.array(current_volley)
        volleys.append(
            {
                "event_indices": event_indices,
                "start_sample": int(event_times[event_indices[0]]),
                "end_sample": int(event_times[event_indices[-1]]),
                "n_events": len(event_indices),
                "start_time_s": event_times[event_indices[0]] / sampling_rate,
                "end_time_s": event_times[event_indices[-1]] / sampling_rate,
            }
        )

    return volleys


def count_volleys_in_files(h5_files, method, min_pulses, max_isi_ms):
    """Count volleys across files for a given detection method."""
    total = 0
    for h5_file in h5_files:
        if method == "waveform":
            event_times, sampling_rate = extract_waveform_event_times(h5_file)
            if event_times is None:
                continue
        else:
            pulse_centers, _, metadata = load_raw_pulse_data(h5_file)
            if pulse_centers is None or len(pulse_centers) == 0:
                continue
            sampling_rate = metadata[0]
            event_times = pulse_centers

        total += len(
            detect_volleys(
                event_times,
                sampling_rate,
                min_pulses=min_pulses,
                max_isi_ms=max_isi_ms,
            )
        )
    return total


def resolve_volley_detection_strategy(h5_files):
    """
    Try detection strategies in order and return the first one that finds volleys.

    Step 1: waveform sub-peaks (>=5 events, ISI <= 2 ms)
    Step 2: pulse centers fallback (>=3 events, ISI <= 5 ms)
    """
    strategies = [
        {
            "step": 1,
            "method": "waveform",
            "min_pulses": VOLLEY_MIN_PULSES,
            "max_isi_ms": VOLLEY_MAX_ISI_MS,
            "label": (
                f"waveform sub-peaks (>={VOLLEY_MIN_PULSES} events, "
                f"ISI <= {VOLLEY_MAX_ISI_MS:g} ms)"
            ),
        },
        {
            "step": 2,
            "method": "centers",
            "min_pulses": VOLLEY_FALLBACK_MIN_PULSES,
            "max_isi_ms": VOLLEY_FALLBACK_MAX_ISI_MS,
            "label": (
                f"pulse centers fallback (>={VOLLEY_FALLBACK_MIN_PULSES} events, "
                f"ISI <= {VOLLEY_FALLBACK_MAX_ISI_MS:g} ms)"
            ),
        },
    ]

    print("\nVolley detection strategy search:")
    print("-" * 70)
    for strategy in strategies:
        count = count_volleys_in_files(
            h5_files,
            strategy["method"],
            strategy["min_pulses"],
            strategy["max_isi_ms"],
        )
        print(
            f"  Step {strategy['step']}: {strategy['label']} -> {count} volleys"
        )
        if count > 0:
            print(f"  Using step {strategy['step']} for pulse-type analysis.")
            return strategy

    print("  No volleys found with any strategy.")
    return None


def detect_file_volleys(h5_file, strategy):
    """Detect volleys in one file using the selected strategy."""
    if strategy["method"] == "waveform":
        event_times, sampling_rate = extract_waveform_event_times(h5_file)
        if event_times is None:
            return [], None
    else:
        pulse_centers, _, metadata = load_raw_pulse_data(h5_file)
        if pulse_centers is None or len(pulse_centers) == 0:
            return [], None
        sampling_rate = metadata[0]
        event_times = pulse_centers

    volleys = detect_volleys(
        event_times,
        sampling_rate,
        min_pulses=strategy["min_pulses"],
        max_isi_ms=strategy["max_isi_ms"],
    )
    return volleys, sampling_rate


def analyze_pulses_around_volleys(
    pulse_centers,
    pulse_markers,
    volleys,
    sampling_rate,
    window_s=VOLLEY_CONTEXT_WINDOW_S,
):
    """
    Analyze which pulse types occur within, before, and after volleys.

    Args:
        pulse_centers (np.array): All pulse center sample indices
        pulse_markers (dict): Pulse type markers (double, wide, fat)
        volleys (list): List of detected volleys
        sampling_rate (float): Sampling rate in Hz
        window_s (float): Time window before/after volley in seconds

    Returns:
        dict: Statistics on pulse types within/before/after volleys and baseline
    """
    window_samples = int(window_s * sampling_rate)
    pulse_types = ["double", "wide", "fat"]

    stats_dict = {
        "total_volleys": len(volleys),
        "within": {ptype: 0 for ptype in pulse_types},
        "within_total": 0,
        "before": {ptype: 0 for ptype in pulse_types},
        "after": {ptype: 0 for ptype in pulse_types},
        "before_total": 0,
        "after_total": 0,
        "baseline": {ptype: 0 for ptype in pulse_types},
        "baseline_total": len(pulse_centers),
    }

    for ptype in pulse_types:
        if ptype in pulse_markers:
            stats_dict["baseline"][ptype] = int(np.sum(pulse_markers[ptype]))

    for volley in volleys:
        start_sample = volley["start_sample"]
        end_sample = volley["end_sample"]

        before_window = (start_sample - window_samples, start_sample)
        after_window = (end_sample, end_sample + window_samples)

        in_volley = (pulse_centers >= start_sample) & (pulse_centers <= end_sample)
        before_mask = (pulse_centers >= before_window[0]) & (
            pulse_centers < before_window[1]
        )
        after_mask = (pulse_centers >= after_window[0]) & (
            pulse_centers < after_window[1]
        )

        before_mask &= ~in_volley
        after_mask &= ~in_volley

        stats_dict["within_total"] += int(np.sum(in_volley))
        stats_dict["before_total"] += int(np.sum(before_mask))
        stats_dict["after_total"] += int(np.sum(after_mask))

        for ptype in pulse_types:
            if ptype not in pulse_markers:
                continue
            marker = pulse_markers[ptype]
            stats_dict["within"][ptype] += int(np.sum(marker[in_volley]))
            stats_dict["before"][ptype] += int(np.sum(marker[before_mask]))
            stats_dict["after"][ptype] += int(np.sum(marker[after_mask]))

    return stats_dict


def compare_volley_pulse_type_enrichment(all_pulse_stats):
    """
    Compare pulse-type proportions in volleys vs baseline using two-proportion z-tests.

    Returns:
        dict: Enrichment statistics per pulse type and context (within/before/after)
    """
    results = {}
    baseline_total = all_pulse_stats["baseline_total"]
    baseline_counts = all_pulse_stats["baseline"]

    for context in ["within", "before", "after"]:
        context_total_key = f"{context}_total"
        context_total = all_pulse_stats[context_total_key]
        context_counts = all_pulse_stats[context]

        for ptype in ["double", "wide", "fat"]:
            baseline_n = baseline_counts[ptype]
            context_n = context_counts[ptype]

            if baseline_total == 0 or context_total == 0:
                z_stat, p_value = np.nan, np.nan
            else:
                p_pool = (baseline_n + context_n) / (baseline_total + context_total)
                se = np.sqrt(
                    p_pool * (1 - p_pool) * (1 / baseline_total + 1 / context_total)
                )
                if se == 0:
                    z_stat, p_value = np.nan, np.nan
                else:
                    p_baseline = baseline_n / baseline_total
                    p_context = context_n / context_total
                    z_stat = (p_context - p_baseline) / se
                    p_value = 2 * stats.norm.sf(abs(z_stat))

            results[(context, ptype)] = {
                "baseline_pct": 100 * baseline_n / baseline_total
                if baseline_total
                else np.nan,
                "context_pct": 100 * context_n / context_total if context_total else np.nan,
                "z_stat": z_stat,
                "p_value": p_value,
            }

    return results


def print_volley_analysis(volley_stats, strategy=None):
    """Print formatted volley analysis results."""
    print(f"\n{'=' * 70}")
    print("HIGH VOLTAGE VOLLEY ANALYSIS")
    print(f"{'=' * 70}")

    if strategy:
        print(f"\nDetection method: {strategy['label']}")

    print("\nVolley Detection Summary:")
    print(f"  Total volleys detected: {volley_stats['total_volleys']}")

    if volley_stats["total_volleys"] == 0:
        print("  No volleys detected in the data.")
        return

    baseline_total = volley_stats["baseline_total"]
    print(f"\nBaseline pulse composition (all detected pulses, n={baseline_total}):")
    print("-" * 70)
    for pulse_type in ["double", "wide", "fat"]:
        count = volley_stats["baseline"][pulse_type]
        pct = 100 * count / baseline_total if baseline_total else 0
        print(f"  {pulse_type.upper():6}: {count:5d} pulses ({pct:5.1f}%)")

    enrichment = compare_volley_pulse_type_enrichment(volley_stats)
    contexts = [
        ("within", "WITHIN volleys", "within_total"),
        (
            "before",
            f"BEFORE volleys ({VOLLEY_CONTEXT_WINDOW_S:.0f}s window)",
            "before_total",
        ),
        (
            "after",
            f"AFTER volleys ({VOLLEY_CONTEXT_WINDOW_S:.0f}s window)",
            "after_total",
        ),
    ]

    for context_key, context_label, total_key in contexts:
        total = volley_stats[total_key]
        print(f"\nPulse Types {context_label} (n={total}):")
        print("-" * 70)
        if total == 0:
            print("  No pulses in this context.")
            continue

        for pulse_type in ["double", "wide", "fat"]:
            count = volley_stats[context_key][pulse_type]
            pct = 100 * count / total
            enrich = enrichment[(context_key, pulse_type)]
            sig = ""
            if not np.isnan(enrich["p_value"]) and enrich["p_value"] < 0.05:
                direction = "enriched" if enrich["context_pct"] > enrich["baseline_pct"] else "depleted"
                sig = f" * {direction} vs baseline (p={enrich['p_value']:.4f})"
            print(
                f"  {pulse_type.upper():6}: {count:5d} pulses ({pct:5.1f}%)"
                f" | baseline {enrich['baseline_pct']:5.1f}%{sig}"
            )


#################################
############# AREA ANALYSIS #############
#################################


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
        "bright": {"double": 0, "wide": 0, "fat": 0, "total": 0, "n_files": 0},
        "dark": {"double": 0, "wide": 0, "fat": 0, "total": 0, "n_files": 0},
        "unlabeled": {"double": 0, "wide": 0, "fat": 0, "total": 0, "n_files": 0},
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
            for ptype in ["double", "wide", "fat"]:
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

        for ptype in ["double", "wide", "fat"]:
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
        for ptype in ["double", "wide", "fat"]:
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
        for ptype in ["double", "wide", "fat"]:
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


#################################
############# MAIN #############
#################################


def main():
    """Run the complete correlation analysis."""
    print("\n" + "=" * 70)
    print("CORRELATING EEL ACTIVITY WITH ENVIRONMENTAL PARAMETERS")
    print("=" * 70 + "\n")

    # Load environmental data
    print("Step 1: Loading environmental data...")
    env_data = load_environmental_data()

    daily_env = aggregate_daily_environmental(env_data)
    monthly_env = aggregate_monthly_environmental(env_data)

    # Dictionary to store all correlations for comparison
    all_correlations = {}
    all_pulses_data = {}

    # Process each pulse type
    for pulse_type, pulse_config in PULSE_TYPES.items():
        print(f"\n{'─' * 70}")
        print(f"Step 2: Loading {pulse_type} pulse data...")
        pulse_rate_dict, metadata, _ = load_pulse_data(pulse_type)

        print("Step 3: Aligning pulse data with environmental data...")
        data_daily = align_daily_data(pulse_rate_dict, metadata, daily_env)
        data_monthly = align_monthly_data(pulse_rate_dict, metadata, monthly_env)

        if len(data_monthly) > 3:
            print("Step 4: Calculating correlations...")
            correlations = calculate_correlations(data_daily, data_monthly)
            print_correlation_summary(correlations, pulse_config["label"])

            # Store correlations for later comparison
            all_correlations[pulse_type] = correlations

            # Store all pulses data for detailed analysis
            if pulse_type == "all":
                all_pulses_data = {
                    "daily": data_daily,
                    "monthly": data_monthly,
                    "correlations": correlations,
                }

            print("Step 5: Creating visualizations...")
            plot_timeseries(data_monthly, pulse_config["label"])
            plot_correlations(data_daily, data_monthly, pulse_config["label"])
        else:
            print(f"Skipping {pulse_type}: insufficient aligned data points")

    # Detailed analysis for all pulses
    if all_pulses_data:
        print_all_pulses_detailed_summary(
            all_pulses_data["correlations"],
            all_pulses_data["daily"],
            all_pulses_data["monthly"],
            all_correlations,
        )
        compare_pulse_types_contribution(all_correlations)

    # Volley detection and analysis
    print(f"\n{'=' * 70}")
    print("Step 6: Detecting high-voltage volleys and pulse associations...")
    print(f"  Pulse-type context window: +/- {VOLLEY_CONTEXT_WINDOW_S:.0f} s")
    print(f"{'=' * 70}")

    h5_root = Path(
        "/home/eisele/wrk/mscthesis/data/raw/eels-mfn2021_dummy_pulses_redetected"
    )
    h5_files = sorted(h5_root.glob("**/*.h5"))

    if len(h5_files) > 0:
        strategy = resolve_volley_detection_strategy(h5_files)

        all_pulse_stats = {
            "total_volleys": 0,
            "within": {"double": 0, "wide": 0, "fat": 0},
            "within_total": 0,
            "before": {"double": 0, "wide": 0, "fat": 0},
            "after": {"double": 0, "wide": 0, "fat": 0},
            "before_total": 0,
            "after_total": 0,
            "baseline": {"double": 0, "wide": 0, "fat": 0},
            "baseline_total": 0,
        }

        if strategy is not None:
            for h5_file in h5_files:
                pulse_centers, pulse_markers, metadata = load_raw_pulse_data(h5_file)
                if pulse_centers is None or len(pulse_centers) == 0:
                    continue

                volleys, sampling_rate = detect_file_volleys(h5_file, strategy)
                if sampling_rate is None:
                    sampling_rate = metadata[0]

                if len(volleys) > 0:
                    print(
                        f"  {h5_file.name}: {len(pulse_centers)} pulses, "
                        f"{len(volleys)} volleys"
                    )

                if volleys and pulse_markers:
                    stats = analyze_pulses_around_volleys(
                        pulse_centers,
                        pulse_markers,
                        volleys,
                        sampling_rate,
                    )

                    all_pulse_stats["total_volleys"] += stats["total_volleys"]
                    all_pulse_stats["within_total"] += stats["within_total"]
                    all_pulse_stats["before_total"] += stats["before_total"]
                    all_pulse_stats["after_total"] += stats["after_total"]
                    all_pulse_stats["baseline_total"] += stats["baseline_total"]
                    for context in ["within", "before", "after", "baseline"]:
                        for ptype in ["double", "wide", "fat"]:
                            all_pulse_stats[context][ptype] += stats[context][ptype]

        print_volley_analysis(all_pulse_stats, strategy=strategy)

        print(f"\n{'=' * 70}")
        print("Step 7: Comparing pulse shapes in bright vs dark tank areas...")
        print(f"{'=' * 70}")
        area_stats = accumulate_area_pulse_stats(h5_files)
        print_area_pulse_analysis(area_stats)
    else:
        print("No .h5 files found. Skipping volley and area analysis.")

    print(f"\n{'=' * 70}")
    print(f"Analysis complete! Output saved to: {OUTPUT_DIR}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()

# %%
