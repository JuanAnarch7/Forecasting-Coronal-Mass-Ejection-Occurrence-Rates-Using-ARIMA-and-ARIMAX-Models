# Scripts Directory

This folder contains the forecasting scripts used to predict CME occurrence rates using ARIMA/ARIMAX time series models with sunspot numbers as exogenous predictors.

All scripts can be executed independently, provided the required input datasets are available in the `data/processed/` directory.

---

## Analysis Organization

Scripts are provided for forecasting analyses performed at two temporal resolutions:

- **Annual temporal resolution** → ARIMA vs ARIMAX models
- **Monthly temporal resolution** → SARIMA vs SARIMAX models (with seasonal components)

Each script performs a complete forecasting pipeline, including:
- Stationarity diagnostics (ADF test)
- Serial correlation testing (Ljung-Box test)
- Automatic model order selection
- Train/test split validation (80/20)
- Out-of-sample forecasting
- Residual diagnostics
- Publication-quality visualization

---

## Inputs

Scripts expect processed CME and sunspot datasets as described in `data/README.md`:

**Required files:**
- `data/datos_procesados_2025_11_30.csv` - Processed CME catalog
- `SN_y_tot_V2.0.txt` - Annual sunspot numbers (for annual scripts)
- `SN_m_tot_V2.0.txt` - Monthly sunspot numbers (for monthly scripts)

---

## Script Types and Outputs

### Forecasting Scripts

#### `annual_forecasting.py`
**Purpose:** Compare ARIMA vs ARIMAX models for annual CME forecasting

**Configuration parameters** (editable at top of script):
```python
MIN_SPEED = 1000      # Minimum CME speed (km/s)
MIN_WIDTH = 0         # Minimum angular width (degrees)
MAX_WIDTH = 360       # Maximum angular width (degrees)
YEAR_START = 1996     # Start year
YEAR_END = 2024       # End year
TRAIN_END_YEAR = 2019 # Last year of training set
TEST_START_YEAR = 2020 # First year of test set
```

**Outputs** (saved to `../results/`):
1. **`figures/acf_pacf_annual_analysis.pdf`**
   - ACF (Autocorrelation Function) plot
   - PACF (Partial Autocorrelation Function) plot
   - Used to justify model order selection

2. **`figures/forecast_comparison_annual_with_validation.pdf`**
   - Main publication figure
   - Shows observed vs predicted CME counts
   - Displays train/test split
   - Includes performance metrics (R², RMSE)

3. **`figures/residual_diagnostics_annual.pdf`**
   - 4-panel diagnostic plot
   - Time series of residuals (ARIMA and ARIMAX)
   - Histogram of residuals (both models)

4. **`tables/forecast_results_annual_with_validation.csv`**
   - Year-by-year numerical results
   - Columns: Year, CMEs_observed, ARIMA_fitted, ARIMAX_fitted, residuals, sunspot_number, Dataset (Train/Test)

5. **`metrics/model_metrics_summary_annual.txt`**
   - Complete statistical summary
   - Training and test metrics (RMSE, MAE, R²)
   - ADF test results
   - Ljung-Box test results
   - Model comparison and recommendation

**Execution:**
```bash
cd scripts
python annual_forecasting.py
```

---

#### `monthly_forecasting.py`
**Purpose:** Compare SARIMA vs SARIMAX models for monthly CME forecasting

**Configuration parameters:**
```python
MIN_SPEED = 1000      # Minimum CME speed (km/s)
MIN_WIDTH = 0         # Minimum angular width (degrees)
MAX_WIDTH = 360       # Maximum angular width (degrees)
YEAR_START = 1996     # Start year
YEAR_END = 2025       # End year (includes partial year)
TRAIN_END = '2019-02' # Last month of training set
TEST_START = '2019-03' # First month of test set
```

**Outputs** (saved to `../results/`):
1. **`figures/acf_pacf_monthly_analysis.pdf`**
   - ACF/PACF plots for monthly data
   - Lags up to 40 months shown

2. **`figures/forecast_comparison_monthly_with_validation.pdf`**
   - Monthly forecast visualization
   - Seasonal patterns visible
   - Performance metrics displayed

3. **`figures/residual_diagnostics_monthly.pdf`**
   - Residual analysis for SARIMA and SARIMAX

4. **`tables/forecast_results_monthly_with_validation.csv`**
   - Month-by-month predictions
   - Columns: Date, YearMonth, CMEs_observed, SARIMA_fitted, SARIMAX_fitted, residuals, sunspot_number, Dataset

5. **`metrics/model_metrics_summary_monthly.txt`**
   - Statistical summary for monthly models
   - Seasonal component analysis

**Execution:**
```bash
cd scripts
python monthly_forecasting.py
```

---

## Analyzing Different CME Subpopulations

To analyze different CME types, modify the filter parameters at the top of each script:

### All CMEs
```python
MIN_SPEED = 0
MIN_WIDTH, MAX_WIDTH = 0, 360
```

### Fast CMEs (V ≥ 1000 km/s)
```python
MIN_SPEED = 1000
MIN_WIDTH, MAX_WIDTH = 0, 360
```

### Halo CMEs (360° width)
```python
MIN_SPEED = 0
MIN_WIDTH, MAX_WIDTH = 360, 360
```

### Partial Halo CMEs (120-360°)
```python
MIN_SPEED = 0
MIN_WIDTH, MAX_WIDTH = 120, 360
```

### Fast Partial Halo CMEs
```python
MIN_SPEED = 450
MIN_WIDTH, MAX_WIDTH = 20, 120
```

**Note:** After changing filters, rename output files to avoid overwriting previous results.

---

## Configuration Notes

### Modifying the Analysis Period

Both scripts allow modification of the temporal period:

**In `annual_forecasting.py`:**
```python
YEAR_START, YEAR_END = 1996, 2024
TRAIN_END_YEAR = 2019  # 80% of data
TEST_START_YEAR = 2020  # 20% of data
```

**In `monthly_forecasting.py`:**
```python
YEAR_START, YEAR_END = 1996, 2025
TRAIN_END = '2019-02'   # Format: 'YYYY-MM'
TEST_START = '2019-03'  # Format: 'YYYY-MM'
```

### Customizing Output Filenames

Output files are automatically named based on content type. To avoid overwriting results from different CME subpopulations, you can:

1. **Option A:** Create separate output directories:
```python
results_dir = '../results/fast_cmes/'
```

2. **Option B:** Add suffix to filenames:
```python
output_prefix = 'fast_cmes'
plt.savefig(f'{output_prefix}_forecast_comparison_annual.pdf')
```

3. **Option C:** Run scripts from different directories and organize results manually

---

## Statistical Tests Performed

### Preliminary Diagnostics

1. **Augmented Dickey-Fuller (ADF) Test**
   - Tests for stationarity
   - H₀: Series has a unit root (non-stationary)
   - If p < 0.05 → Series is stationary
   - Determines differencing order (d)

2. **Ljung-Box Test (Initial)**
   - Tests for serial correlation
   - H₀: No autocorrelation (white noise)
   - If p < 0.05 → Autocorrelation present (ARIMA appropriate)

3. **Pearson Correlation**
   - Measures linear relationship between SSN and CME counts
   - Reported with p-value for significance

### Model Validation

1. **Ljung-Box Test (Residuals)**
   - Tests if residuals are white noise
   - Desired: p > 0.05 (no remaining autocorrelation)
   - Indicates model has captured all patterns

2. **Out-of-Sample Metrics**
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - R² (Coefficient of Determination)
   - AIC (Akaike Information Criterion)

---

## Model Selection

Scripts use **automatic model order selection** via `pmdarima.auto_arima`:

**Annual models:**
- max_p = 5, max_q = 5 (or adaptive based on sample size)
- max_d = 2
- Selection criterion: AIC

**Monthly models:**
- max_p = 5, max_q = 5, max_d = 2
- max_P = 2, max_Q = 2, max_D = 1
- Seasonal period: m = 12
- Selection criterion: AIC

**Model orders are automatically determined and reported in output.**

---

## Console Output

When executed, scripts provide detailed progress information:

```
================================================================================
ARIMA vs ARIMAX MODEL COMPARISON WITH OUT-OF-SAMPLE VALIDATION
================================================================================

Train period: 1996-2019
Test period:  2020-2024

[1/9] Loading CME dataset...
[2/9] Filtering CME events...
   Total CMEs: 32,000 → Filtered CMEs: 1,234
   Years with zero CMEs: 3

[3/9] Loading sunspot numbers...
   Pearson correlation (SSN vs CME): r = 0.8234 (p = 1.23e-05)

...

┌────────────────────────────────────────────────────────────────────┐
│ TEST SET PERFORMANCE (Out-of-Sample) ⭐ KEY METRICS               │
├────────────────────────────────────────────────────────────────────┤
│ Metric              │ ARIMA      │ ARIMAX     │ Improvement      │
├────────────────────────────────────────────────────────────────────┤
│ RMSE (CMEs/year)    │     52.345 │     44.123 │ ✓      +15.71%  │
│ MAE (CMEs/year)     │     41.234 │     35.678 │ ✓      +13.47%  │
│ R² (coefficient)    │     0.6234 │     0.7123 │ ✓      +14.26%  │
└────────────────────────────────────────────────────────────────────┘

 WINNER: ARIMAX
   Reason: Lower forecast error AND better model fit (lower AIC)
```

---

## Troubleshooting

### Common Issues

**Problem:** `FileNotFoundError: datos_procesados_2025_11_30.csv`
```
Solution: Run data processing first
cd ../data_processing
python Lecture_data_CME.py
```

**Problem:** Empty plots or very poor R² values
```
Cause: CME subpopulation is too rare (many zero years)
Solution: This is valid! Document as "sparse occurrence pattern"
Consider using monthly resolution for better sample size
```

**Problem:** Warning about convergence
```
Cause: Difficult optimization due to data characteristics
Solution: Usually safe to ignore if residual diagnostics look good
Check Ljung-Box p-value > 0.05 for validation
```

**Problem:** Different results each run
```
Cause: auto_arima may find different local optima
Solution: Normal behavior. Focus on test set metrics
Results should be qualitatively similar
```
## Execution Notes

- Scripts run independently (no dependencies between them)
- Execution time: ~1-5 minutes per script (depending on data size)
- Results are deterministic for ARIMA/ARIMAX (given same data and constraints)
- SARIMA/SARIMAX may show minor variations due to optimization

**Recommended workflow:**
1. Run annual script first (faster, easier to interpret)
2. Validate results make sense
3. Run monthly script for higher resolution analysis
4. Compare results between temporal resolutions

