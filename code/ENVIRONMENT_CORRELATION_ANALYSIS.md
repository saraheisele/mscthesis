# Eel Activity & Environmental Correlation Analysis

## Overview
This analysis investigates correlations between eel pulse activity (measured in Hz) and water environmental parameters (temperature and conductivity) over the recording period.

## Data Sources

### Environmental Data
- **Files**: 4 Excel files covering different time periods
  - `Verlauf-relLeitwert_220124-220812.xlsx`: 2022-01-24 to 2022-08-12
  - `Verlauf-relLeitwert_220810-230104.xlsx`: 2022-08-10 to 2023-01-04
  - `Verlauf-relLeitwert_2024.xlsx`: 2024 measurements
  - `Verlauf-relLeitwert_2025.xlsx`: 2025 measurements
- **Total Records**: 37 environmental measurements
- **Date Range**: 2022-02-04 to 2026-01-05
- **Temperature Range**: 25.0°C to 27.0°C (relatively stable)
- **Conductivity Range**: 178 to 428 µS/cm

### Pulse Activity Data
- **Source**: Preprocessing pipeline outputs
- **Pulse Types Analyzed**:
  - All pulses (total activity)
  - Double pulses
  - Wide pulses
  - Fat pulses
- **Date Range**: 2023-11-01 to 2026-03-01
- **Overlap with Env Data**: 2023-11 to 2026-03
- **Timescales**: Daily, Monthly (month_since_start)

## Alignment Strategy

### Daily Level (13 matched records)
- Daily pulse rates (Hz) aggregated from day-of-year bins
- Daily environmental averages (when multiple measurements on same day)
- Focuses on short-term variability

### Monthly Level (14 matched records)
- Monthly pulse rates using "month_since_start" bins
- Monthly environmental averages
- Focuses on longer-term trends

## Key Findings

### Significant Correlations (p < 0.05)

1. **Double Pulses - Conductivity (Daily)**
   - Spearman ρ = 0.644, p = 0.0175 ✓
   - **Interpretation**: Double pulse rate increases with water conductivity (daily level)
   - Pearson r = 0.413, p = 0.1602 (non-significant linear correlation)
   - Suggests non-linear or rank-order relationship

2. **Wide Pulses - Conductivity (Monthly)**
   - Spearman ρ = -0.640, p = 0.0138 ✓
   - **Interpretation**: Wide pulse rate decreases with increasing conductivity (monthly level)
   - Pearson r = -0.390, p = 0.1675 (non-significant linear)

3. **Fat Pulses - Conductivity (Monthly)**
   - Spearman ρ = -0.626, p = 0.0165 ✓
   - **Interpretation**: Fat pulse rate decreases with increasing conductivity (monthly level)
   - Pearson r = -0.258, p = 0.3728 (non-significant linear)

### Non-Significant Correlations
- **Temperature**: No significant correlations at any level for any pulse type
  - All p-values > 0.40
  - Likely due to narrow temperature range (only 2°C variation: 25-27°C)
- **All Pulses & Conductivity**: No significant correlations
- **Double Pulses & Temperature**: No significant correlations
- **Wide/Fat Pulses & Temperature**: No significant correlations

## Statistical Notes

- **Sample Sizes**: Limited (n=13-14), which reduces statistical power
- **Statistical Tests**: Both Pearson (linear) and Spearman (rank-order) correlations reported
- **Significance Level**: p < 0.05 marked with *
- **Environmental Variance**: Temperature shows very little variation (2°C range), limiting correlation potential

## Biological Interpretation

1. **Conductivity Sensitivity**: Eels show differential sensitivity to conductivity depending on pulse type:
   - Double pulses increase with conductivity
   - Wide and fat pulses decrease with conductivity
   - Suggests different physiological or behavioral functions of pulse types

2. **Temperature**: No significant effects likely due to:
   - Narrow temperature range in data (25-27°C)
   - Possible temperature acclimation
   - Activity may be controlled by factors other than absolute temperature

3. **Timescale Dependence**: Significant correlations mostly at monthly (not daily) level for wide/fat pulses:
   - Suggests long-term adaptation rather than direct stimulus-response
   - Short-term conductivity changes may not directly affect pulse rate

## Output Files

### Visualizations
- `timeseries_*.png`: Time series plots showing all three variables
  - Top: Pulse rate (Hz) over time
  - Middle: Water temperature (°C)
  - Bottom: Conductivity (µS/cm)

- `correlations_*.png`: Scatter plot matrices showing:
  - Daily pulse rate vs temperature
  - Daily pulse rate vs conductivity
  - Monthly pulse rate vs temperature
  - Monthly pulse rate vs conductivity
  - Temperature vs conductivity relationship
  - Bubble plot: pulse rate vs temp (point size = conductivity)

### Directory
All outputs saved to: `/home/eisele/wrk/mscthesis/data/processed/environment_correlation/`

## Recommendations

1. **Increase Sample Size**: Current n=13-14 limits power. Aim for 30+ data points per correlation.

2. **Temperature Control**: The narrow temperature range limits analysis. Consider:
   - Analyzing relative changes rather than absolute values
   - Looking for threshold effects
   - Seasonal analysis if temperature varies more in other seasons

3. **Conductivity Analysis**: Clear patterns emerge with conductivity:
   - Investigate what causes conductivity changes (evaporation, water replacement, feeding effects)
   - Consider lag effects (pulse changes after conductivity changes)
   - Examine if conductivity reflects water quality changes

4. **Additional Variables**: Consider analyzing:
   - Water pH
   - Dissolved oxygen
   - Photoperiod (day length)
   - Circadian patterns

5. **Pulse Type Specificity**: Different pulse types show opposite patterns with conductivity:
   - Investigate physiological basis for these differences
   - May reflect different functional roles (communication, hunting, aggression)

## Script Details

**File**: `correlate_activity_with_environment.py`

**Functions**:
- `load_environmental_data()`: Consolidates all Excel files
- `aggregate_daily/monthly_environmental()`: Creates temporal averages
- `load_pulse_data()`: Loads preprocessing outputs
- `align_daily/monthly_data()`: Matches pulse data with environment data
- `calculate_correlations()`: Computes Pearson & Spearman correlations
- `plot_timeseries()`: Generates time series visualizations
- `plot_correlations()`: Generates scatter plots with statistics

**How to Modify**:
- Change `PULSE_TYPES` to analyze different pulse categories
- Adjust alignment strategy in `align_*` functions for different time shifts
- Add new correlation types or statistical tests in `calculate_correlations()`
